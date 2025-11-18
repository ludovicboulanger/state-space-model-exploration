from os import environ
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))


from torch import set_flush_denormal
from config import TrainingConfig


from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from modules.classification_module import ClassificationModule
from utils.registries import DatasetRegistry


def _run_test_on_pdm(run: str, pdm_factor: int) -> float:
    config = TrainingConfig.from_json(Path(run + "/training_config.json"))
    running_on_slurm = environ.get("SLURM_JOB_ID", None) is not None

    if not running_on_slurm:
        config.save_dir = str(Path(run).parent)
        config.data_root = str(Path(__file__).parents[1] / "data")
    else:
        config.data_root = environ["SLURM_TMPDIR"] + "/data/"

    testing_dataset = DatasetRegistry.instantiate(config, split="validation")
    testing_dataset.data_encoding = "pdm"
    testing_dataset.upsampling_factor = pdm_factor
    testing_loader = DataLoader(
        testing_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
    )
    trainer = Trainer(
        deterministic=True,
        log_every_n_steps=100,
        logger=CSVLogger(
            save_dir=Path(config.save_dir) / config.run_id,
            name="logs",
            version=f"valid-{testing_dataset.data_encoding}-{str(pdm_factor).replace('.', '')}",
        ),
    )

    ckpt_file = Path(config.save_dir) / config.run_id / "best_checkpoint.ckpt"
    ssm = ClassificationModule.load_from_checkpoint(ckpt_file, config=config, output_dim=testing_dataset.num_labels)

    ssm.update_sampling_rate(pdm_factor / config.upsampling_factor)

    trainer.test(model=ssm, dataloaders=testing_loader)
    return trainer.callback_metrics.get("test_accuracy")  # type: ignore


def _get_run_loc() -> str:
    task_id = environ.get("SLURM_ARRAY_TASK_ID", "0")
    base_path = r"/home/ludoboul/projects/def-seanwood/ludoboul/training-runs/ssm-speech-processing/google_speech_commands_small"
    if task_id == "0":
        return base_path + "/52424530_0"
    elif task_id == "1":
        return base_path + "/52424530_1"
    elif task_id == "2":
        return base_path + "/52424530_2"
    elif task_id == "3":
        return base_path + "/52424530_3"
    elif task_id == "4":
        return base_path + "/52424530_4"
    elif task_id == "5":
        return base_path + "/52424530_5"
    else:
        raise ValueError(f"Unknown Task ID : {task_id}")


def main() -> None:
    if environ.get("SLURM_JOB_ID", None) is not None:
        run = _get_run_loc()
    else:
        run = r"/home/ludovic/workspace/ssm-speech-processing/training-runs/local/ssm-speech-processing/google_speech_commands_small/821056"
    pdm_factors_to_test = [8, 16, 32, 64]
    for factor in pdm_factors_to_test:
        _run_test_on_pdm(run, factor)


if __name__ == "__main__":
    seed_everything(3221)
    set_flush_denormal(True)
    main()
