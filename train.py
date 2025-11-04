from os import environ

from pathlib import Path
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.plugins.environments import SLURMEnvironment  # type: ignore
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from config import ConfigParser, TrainingConfig
from modules.classification_module import ClassificationModule
from modules.regression_module import RegressionModule
from utils.registries import DatasetRegistry


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

    training_dataset = DatasetRegistry.instantiate(config, "training")
    validation_dataset = DatasetRegistry.instantiate(config, "validation")

    training_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=_get_cpu_count_per_node(),
        pin_memory=use_ddp,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1 if config.task == "regression" else config.batch_size,
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

    if config.task == "classification":
        ssm = ClassificationModule(config, training_dataset.num_labels)
    else:
        ssm = RegressionModule(config, output_dim=training_dataset.num_labels)
    summary = ModelSummary(ssm, max_depth=1000)
    with open(Path(config.save_dir) / config.run_id / "model_summary.txt", "w") as ostream:
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
    testing_dataset = DatasetRegistry.instantiate(config, split="testing")
    testing_loader = DataLoader(
        testing_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=_get_cpu_count_per_node(),
        pin_memory=False,
    )
    trainer = Trainer(
        deterministic=True,
        limit_test_batches=10,
        log_every_n_steps=100,
        logger=CSVLogger(save_dir=Path(config.save_dir) / config.run_id, name="logs", version="test"),
    )

    if config.task == "classification":
        ssm = ClassificationModule(config, testing_dataset.num_labels)
    else:
        ssm = RegressionModule(config, output_dim=testing_dataset.num_labels)

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
    run_to_test = Path(
        "/home/ludovic/workspace/ssm-speech-processing/training-runs/local/ssm-speech-processing/voicebank_demand/245514"
    )
    config = TrainingConfig.from_json(run_to_test / "training_config.json")
    test_model(config)


if __name__ == "__main__":
    test()
