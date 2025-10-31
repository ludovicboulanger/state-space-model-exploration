from dataclasses import dataclass
from os import environ

from pathlib import Path
from typing import Any, Dict, List, Tuple
from lightning import Trainer, LightningModule
from lightning.pytorch import seed_everything
from lightning.pytorch.plugins.environments import SLURMEnvironment  # type: ignore
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor, ones_like
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.nn import ModuleList, Linear
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy

from config import ConfigParser, TrainingConfig
from google_speech_commands_dataset_small import SpeechCommandsDatasetSmall
from mnist_dataset import MNISTDataset
from speech_commands_dataset import SpeechCommandsDataset
from s4 import S4Block
from utils import PDMEncoder, Upsampler


@dataclass
class ForwardPassOutput:
    logits: Tensor
    layer_outputs: List[Tensor]
    loss: Tensor
    acc: Tensor


class S4Network(LightningModule):
    def __init__(self, config: TrainingConfig, output_dim: int) -> None:
        super().__init__()
        self._inference_mode = False

        self._encoder = Linear(in_features=1, out_features=config.channel_dim)
        self._encoder.weight.data = ones_like(self._encoder.weight.data)
        self._encoder.weight.requires_grad = False

        self._blocks = ModuleList([])
        for i in range(config.num_layers):
            self._blocks.append(
                S4Block(
                    channels=config.channel_dim,
                    n_ssms=config.num_ssms,
                    state_dim=config.hidden_dim,
                    seq_len=config.seq_len,
                    min_dt=config.min_dt,
                    max_dt=config.max_dt,
                    clip_B=config.clip_B,
                    residual=1 > 0,
                    p_kernel_dropout=config.kernel_dropout_prob,
                    p_block_dropout=config.block_dropout_prob,
                    norm=config.norm if i > 0 else "none",
                    prenorm=config.pre_norm,
                    layer_activation=config.layer_activation,
                    final_activation=config.final_activation,
                )
            )
        self._decoder = Linear(in_features=config.channel_dim, out_features=output_dim)
        self._accuracy = Accuracy(task="multiclass", num_classes=output_dim)
        self._config = config

    def training_step(self, batch: Tuple[Tensor, ...]) -> Tensor:
        out = self._forward_pass(batch)
        self.log(
            name="train_loss",
            value=out.loss.item(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True
        )
        self.log(
            name="train_accuracy",
            value=out.acc.item(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return out.loss

    def validation_step(self, batch: Tuple[Tensor, ...]) -> Tensor:
        out = self._forward_pass(batch)
        self.log(
            name="valid_loss",
            value=out.loss.item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True
        )
        self.log(
            name="valid_accuracy",
            value=out.acc.item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return out.loss

    def test_step(self, batch: Tuple[Tensor, ...]) -> Tensor:
        out = self._forward_pass(batch)
        self.log(
            name="test_accuracy",
            value=out.acc.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return out.loss

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore
        hippo_parameters = []
        nn_parameters = []
        for name, param in self._blocks.named_parameters():
            param_name = name.split(".")[-1]
            if param_name in ["A_real", "A_imag", "_P", "_B", "_C", "_dt"]:
                hippo_parameters.append(param)
            else:
                nn_parameters.append(param)

        optimizer = AdamW(
            params=[
                {"params": hippo_parameters, "lr": min(self._config.lr, 1e-3), "weight_decay": 0.0},
                {"params": nn_parameters, "lr": self._config.lr, "weight_decay": self._config.weight_decay},
            ],
        )
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=200000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }

    def update_sampling_rate(self, sampling_factor: int) -> None:
        for block in self._blocks:
            block._s4_layer._kernel.scale_factor = sampling_factor #type: ignore

    def _forward_pass(
        self, batch: Tuple[Tensor, ...]
    ) -> ForwardPassOutput:
        x, y = batch
        layer_outputs = []

        y_hat = self._encoder(x)
        for block in self._blocks:
            y_hat, layer_out = block(y_hat)
            layer_outputs.append(layer_out)
        y_hat = y_hat.mean(dim=1)
        logits = self._decoder(y_hat)

        loss = cross_entropy(logits, y)
        acc = self._accuracy(logits.argmax(dim=-1), y)
        return ForwardPassOutput(logits=logits, layer_outputs=layer_outputs, loss=loss, acc=acc)


def train_model(config: TrainingConfig) -> float:
    (Path(config.save_dir) / config.run_id).mkdir(parents=True, exist_ok=True)
    config.to_json(Path(config.save_dir) / config.run_id / "training_config.json")

    if config.debug:
        _wait_for_debugger()

    if _get_node_count() > 1 or _get_device_count_per_node() > 1:
        use_ddp = True
        training_strategy = "ddp"
        plugins = [SLURMEnvironment(auto_requeue=False)]
    else:
        use_ddp = False
        training_strategy = "auto"
        plugins = None

    training_dataset = SpeechCommandsDatasetSmall(
        root=config.data_root,
        download=True,
        subset="training",
        data_encoding=config.data_encoding,
        pdm_factor=config.pdm_factor
    )
    validation_dataset = SpeechCommandsDatasetSmall(
        root=config.data_root,
        download=True,
        subset="validation",
        data_encoding=config.data_encoding,
        pdm_factor=config.pdm_factor
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=_get_cpu_count_per_node(),
        pin_memory=use_ddp,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=_get_cpu_count_per_node(),
        pin_memory=use_ddp,
    )
    trainer = Trainer(
        deterministic=True,
        log_every_n_steps=100,
        accumulate_grad_batches=config.accumulate_grad_batches,
        devices=_get_device_count_per_node(),
        num_nodes=_get_node_count(),
        strategy=training_strategy,
        plugins=plugins,  # type: ignore
        max_epochs=config.max_epochs,
        logger=CSVLogger(save_dir=Path(config.save_dir) / config.run_id, name="logs"),
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                dirpath=Path(config.save_dir) / config.run_id,
                filename="last_checkpoint",
                every_n_epochs=1,
                save_on_train_epoch_end=False,
                enable_version_counter=False,
            ),
            ModelCheckpoint(
                dirpath=Path(config.save_dir) / config.run_id,
                filename="best_checkpoint",
                monitor="valid_loss",
                mode="min",
                save_last=False,
                save_on_train_epoch_end=False,
                enable_version_counter=False,
            ),
        ],
    )

    ssm = S4Network(config, output_dim=training_dataset.num_labels)
    summary = ModelSummary(ssm, max_depth=1000)
    with open(
        Path(config.save_dir) / config.run_id / "model_summary.txt", "w"
    ) as ostream:
        ostream.write(str(summary))

    ckpt_file = Path(config.save_dir) / config.run_id / "last_checkpoint.ckpt"
    trainer.fit(
        ssm,
        train_dataloaders=training_loader,
        val_dataloaders=validation_loader,
        ckpt_path=ckpt_file if ckpt_file.exists() else None,
    )

    (Path(config.save_dir) / config.run_id / "training_complete.flag").touch()
    print("Training complete!")
    return trainer.callback_metrics.get("valid_loss")  # type: ignore


def test_model(config: TrainingConfig) -> float:
    pdm_factor = 8
    config.seq_len = pdm_factor * config.seq_len
    testing_dataset = SpeechCommandsDatasetSmall(
        root=config.data_root,
        download=True,
        subset="validation",
        data_encoding="pdm",
        pdm_factor=pdm_factor
    )
    testing_loader = DataLoader(
        testing_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=_get_cpu_count_per_node(),
        pin_memory=False
    )
    trainer = Trainer(
        deterministic=True,
        log_every_n_steps=100,
        logger=CSVLogger(save_dir=Path(config.save_dir) / config.run_id, name="logs", version=f"test-{pdm_factor}"),
    )

    ssm = S4Network(config, output_dim=testing_dataset.num_labels)
    ssm.update_sampling_rate(pdm_factor)

    ckpt_file = Path(config.save_dir) / config.run_id / "best_checkpoint.ckpt"

    trainer.test(model=ssm, dataloaders=testing_loader, ckpt_path=ckpt_file)

    return trainer.callback_metrics.get("test_accuracy")  # type: ignore


def _get_node_count() -> int:
    num_nodes = environ.get("SLURM_NNODES", 1)
    return int(num_nodes)


def _get_device_count_per_node() -> int:
    gpus_per_node = environ.get("SLURM_GPUS_PER_NODE", ":0")
    # TODO: Update so it supports your LOCAL Machine, ie if 0, get cuda_available_devices or something
    if int(gpus_per_node.split(":")[-1]) == 0:
        return _get_cpu_count_per_node()
    return int(gpus_per_node.split(":")[-1])


def _get_cpu_count_per_node() -> int:
    cpus_per_task = environ.get("SLURM_CPUS_PER_TASK", 1)
    return int(cpus_per_task)


def _wait_for_debugger() -> None:
    from debugpy import listen, wait_for_client, breakpoint

    print("Waiting for debugger to attach...")
    listen(5678)
    wait_for_client()
    print("Debugger attached!")
    breakpoint()


def train() -> None:
    from torch.cuda import device_count

    print(f"RANK: {environ.get('SLURM_PROCID', 'N/A')}")
    print(f"LOCAL_RANK: {environ.get('SLURM_LOCALID', 'N/A')}")
    print(f"CUDA_VISIBLE_DEVICES: {environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")
    print(f"Available GPUs: {device_count()}")
    seed_everything(3221)
    config = ConfigParser.from_cli_args()
    train_model(config=config)

def test() -> None:
    run_to_test = Path("/home/ludovic/workspace/ssm-speech-processing/training-runs/local/ssm-speech-processing/google_speech_commands/264405")
    config = TrainingConfig.from_json(run_to_test / "training_config.json")
    test_model(config)


if __name__ == "__main__":
    train()
