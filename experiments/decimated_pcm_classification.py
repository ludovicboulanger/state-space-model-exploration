from os import environ
from pathlib import Path
import sys
from typing import Tuple

from torch import set_flush_denormal


sys.path.append(str(Path(__file__).parents[1]))

from lightning import seed_everything

from config import TrainingConfig
from train import train_model


def _get_decimation_and_encoder() -> Tuple[str, int]:
    task_id = environ.get("SLURM_ARRAY_TASK_ID", "0")
    if task_id == "0":
        return "lmu", 1
    elif task_id == "1":
        return "lmu", 32
    elif task_id == "2":
        return "lmu", 128
    elif task_id == "3":
        return "s4", 1
    elif task_id == "4":
        return "s4", 32
    elif task_id == "5":
        return "s4", 128
    else:
        raise ValueError(f"Unknown Task ID : {task_id}")


def main():
    if environ.get("SLURM_JOB_ID", None):
        save_dir = "/home/ludoboul/projects/def-seanwood/ludoboul/training-runs/ssm-speech-processing/google_speech_commands_small"
        data_root = environ["SLURM_TMPDIR"] + "/data/"
        run_id = environ["SLURM_ARRAY_JOB_ID"] + "_" + environ["SLURM_ARRAY_TASK_ID"]
        batch_size = 8
        encoder, decimation_factor = _get_decimation_and_encoder()

    else:
        encoder = "lmu"
        decimation_factor = 128
        run_id = "821056"
        batch_size = 8
        save_dir = str(
            Path(__file__).parents[1] / "training-runs/local/ssm-speech-processing/google_speech_commands_small"
        )
        data_root = str(Path(__file__).parents[1] / "data")

    config = TrainingConfig()
    config.save_dir = save_dir
    config.data_root = data_root
    config.run_id = run_id
    config.task = "classification"
    config.dataset = "gcs-sm"
    config.data_encoding = "pcm"
    config.upsampling_factor = 1
    config.batch_size = batch_size
    config.accumulate_grad_batches = 2
    config.max_epochs = 40
    config.lr = 1e-2
    config.weight_decay = 0.05
    config.encoder = encoder
    config.encoder_memory_size_seconds = 0.008
    config.decimation_factor = decimation_factor
    config.layer_activation = "gelu"
    config.final_activation = "glu"
    config.norm = "group"
    config.num_layers = 6
    config.hidden_dim = 64
    config.channel_dim = 64
    config.num_ssms = 2
    config.min_dt = 1 / (16_000 / decimation_factor)
    config.max_dt = 1e-1
    config.seq_len = 16_000

    train_model(config)


if __name__ == "__main__":
    seed_everything(3221)
    # Flush really small values. Needed to for computing the LMU impulse response.
    # If it isn't used, the code is just too slow.
    set_flush_denormal(True)
    main()
