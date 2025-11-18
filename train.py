from os import environ

from pathlib import Path
from lightning import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment  # type: ignore
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from config import TrainingConfig
from modules.classification_module import ClassificationModule
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

    # Only Classification is supported for now
    ssm = ClassificationModule(config, training_dataset.num_labels)

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


def _wait_for_debugger() -> None:
    from debugpy import listen, wait_for_client, breakpoint

    print("Waiting for debugger to attach...")
    listen(5678)
    wait_for_client()
    print("Debugger attached!")
    breakpoint()
