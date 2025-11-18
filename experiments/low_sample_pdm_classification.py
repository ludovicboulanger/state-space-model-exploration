from math import floor, log10
from os import environ
from pathlib import Path
import sys

from torch import set_flush_denormal


sys.path.append(str(Path(__file__).parents[1]))

from lightning import seed_everything

from config import TrainingConfig
from train import train_model


def _get_upsampling_factor() -> int:
    task_id = environ.get("SLURM_ARRAY_TASK_ID", "0")
    if task_id == "0":
        return 8
    elif task_id == "1":
        return 16
    elif task_id == "2":
        return 32
    elif task_id == "3":
        return 64
    else:
        raise ValueError(f"Unknown Task ID : {task_id}")


def main():
    if environ.get("SLURM_JOB_ID", None):
        save_dir = "/home/ludoboul/projects/def-seanwood/ludoboul/training-runs/ssm-speech-processing/google_speech_commands_small"
        data_root = environ["SLURM_TMPDIR"] + "/data/"
        run_id = environ["SLURM_ARRAY_JOB_ID"] + "_" + environ["SLURM_ARRAY_TASK_ID"]
        batch_size = 4
        upsampling_factor = _get_upsampling_factor()
    else:
        upsampling_factor = 64
        save_dir = str(
            Path(__file__).parents[1] / "training-runs/local/ssm-speech-processing/google_speech_commands_small"
        )
        data_root = str(Path(__file__).parents[1] / "data")
        run_id = "test-3"
        batch_size = 4

    config = TrainingConfig()
    config.save_dir = save_dir
    config.data_root = data_root
    config.run_id = run_id
    config.task = "classification"
    config.dataset = "gcs-sm"
    config.data_encoding = "pdm"
    config.upsampling_factor = upsampling_factor
    config.batch_size = batch_size
    config.accumulate_grad_batches = 4
    config.max_epochs = 50
    config.lr = 1e-2
    config.weight_decay = 0.05
    config.encoder = "lmu"
    config.encoder_memory_size_samples = 512 * upsampling_factor
    config.decimation_factor = 64
    config.layer_activation = "gelu"
    config.final_activation = "glu"
    config.norm = "group"
    config.num_layers = 6
    config.hidden_dim = 64
    config.channel_dim = 128
    config.num_ssms = 2
    config.min_dt = 10 ** (-1 * floor(log10(16_000 * upsampling_factor)))
    config.max_dt = 1e-1
    config.seq_len = 16_000 * upsampling_factor

    print("Training Model")

    train_model(config)


if __name__ == "__main__":
    seed_everything(3221)
    set_flush_denormal(True)
    main()
