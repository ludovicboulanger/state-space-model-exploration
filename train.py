from os import environ

from pathlib import Path
from typing import Any, Dict, Tuple
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
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import ModuleList, Linear
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy

from config import ConfigParser, TrainingConfig
from google_speech_commands_dataset_small import SpeechCommandsDatasetSmall
from mnist_dataset import MNISTDataset
from speech_commands_dataset import SpeechCommandsDataset
from s4 import S4Block


class S4Network(LightningModule):
    def __init__(self, config: TrainingConfig, output_dim: int) -> None:
        super().__init__()
        self._inference_mode = False
        self._encoder = Linear(in_features=1, out_features=config.channel_dim)
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
                    p_kernel_dropout=config.kernel_dropout_prob,
                    p_block_dropout=config.block_dropout_prob,
                    norm=config.norm,
                    prenorm=config.pre_norm,
                    layer_activation=config.layer_activation,
                    final_activation=config.final_activation,
                )
            )
        self._decoder = Linear(in_features=config.channel_dim, out_features=output_dim)
        self._accuracy = Accuracy(task="multiclass", num_classes=output_dim)
        self._config = config

    @property
    def inference_mode(self) -> bool:
        return self._inference_mode

    @inference_mode.setter
    def inference_mode(self, mode: bool) -> None:
        self._inference_mode = mode
        for block in self._blocks:
            block.inference_mode = mode  # type: ignore

    def training_step(self, batch: Tuple[Tensor, ...]) -> Tensor:
        _, metrics = self._forward_pass(batch)
        self.log(
            name="train_loss",
            value=metrics["loss"].item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            name="train_accuracy",
            value=metrics["acc"].item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return metrics["loss"]

    def validation_step(self, batch: Tuple[Tensor, ...]) -> Tensor:
        _, metrics = self._forward_pass(batch)
        self.log(
            name="valid_loss",
            value=metrics["loss"].item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            name="valid_accuracy",
            value=metrics["acc"].item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return metrics["loss"]

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore
        hippo_parameters = []
        nn_parameters = []
        for name, param in self._blocks.named_parameters():
            param_name = name.split(".")[-1]
            if param_name in ["_L", "_P", "_C", "_B", "_step"]:
                hippo_parameters.append(param)
            else:
                nn_parameters.append(param)

        optimizer = Adam(
            params=[
                {"params": hippo_parameters, "lr": min(self._config.lr, 1e-3)},
                {"params": nn_parameters, "lr": self._config.lr},
            ],
            weight_decay=0,
        )
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=config.lr_decay,
            patience=config.lr_decay_patience,
            threshold=config.lr_delta_threshold,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "valid_loss",
                "strict": True,
            },
        }

    def _forward_pass(
        self, batch: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        x, y = batch

        y_hat = self._encoder(x)
        for block in self._blocks:
            y_hat = block(y_hat)
        y_hat_no_agg = y_hat
        y_hat = y_hat.mean(dim=1)
        logits = self._decoder(y_hat)

        loss = cross_entropy(logits, y)
        acc = self._accuracy(logits.argmax(dim=-1), y)
        return y_hat_no_agg, {"loss": loss, "acc": acc}


def train_model(config: TrainingConfig) -> float:
    (Path(config.save_dir) / config.run_id).mkdir(parents=True, exist_ok=True)
    config.to_json(Path(config.save_dir) / config.run_id / "training_config.json")

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
    )
    validation_dataset = SpeechCommandsDatasetSmall(
        root=config.data_root,
        download=True,
        subset="validation",
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
    """
    EarlyStopping(
        monitor="valid_loss",
        patience=config.early_stop_patience,
        min_delta=config.early_stop_threshold,
    ),
    """
    trainer = Trainer(
        deterministic=True,
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


if __name__ == "__main__":
    from torch.cuda import device_count

    print(f"RANK: {environ.get('SLURM_PROCID', 'N/A')}")
    print(f"LOCAL_RANK: {environ.get('SLURM_LOCALID', 'N/A')}")
    print(f"CUDA_VISIBLE_DEVICES: {environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")
    print(f"Available GPUs: {device_count()}")
    seed_everything(3221)
    config = ConfigParser.from_cli_args()
    train_model(config=config)
