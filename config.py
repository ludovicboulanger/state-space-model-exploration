from __future__ import annotations
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from json import dump, load
from pathlib import Path
from typing import Union


@dataclass
class TrainingConfig:
    # Persistence
    save_dir: str = str(Path(__file__).parent / "training-runs")
    data_root: str = str(Path(__file__).parent / "data")
    run_id: str = "test-run-id"
    smoke_test: bool = False
    # Training Hyperparameters
    max_epochs: int = 100
    batch_size: int = 8
    accumulate_grad_batches: int = 1
    lr: float = 1e-3
    lr_decay: float = 0.8
    lr_decay_patience: int = 3
    lr_delta_threshold: float = 1e-4
    early_stop_patience: int = 10
    early_stop_threshold: float = 1e-4
    weight_decay: float = 5e-2
    # NN Hyperparameters
    num_ssms: int = 1
    num_layers: int = 4
    hidden_dim: int = 8
    channel_dim: int = 8
    min_dt: float = 1e-4
    max_dt: float = 1e-1
    clip_B: float = 2.0
    kernel_dropout_prob: float = 0.0
    block_dropout_prob: float = 0.0
    pre_norm: bool = True
    norm: str = "batch"
    layer_activation: str = "gelu"
    final_activation: str = "glu"
    seq_len: int = 16000

    def __getitem__(self, key: str) -> Union[str, float, int, bool]:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Union[str, float, int, bool]) -> None:
        setattr(self, key, value)

    @staticmethod
    def from_json(path: Path) -> TrainingConfig:
        training_config = TrainingConfig()
        with open(path, "r") as istream:
            config_dict = load(istream)

        for key, value in config_dict.items():
            training_config[key] = value

        return training_config

    def to_json(self, path: Path) -> None:
        with open(path, "w") as ostream:
            dump(asdict(self), ostream)


class ConfigParser:
    def __init__(self) -> None:
        self._default_config: TrainingConfig = TrainingConfig()
        self._run_config: TrainingConfig = TrainingConfig()

    @staticmethod
    def from_cli_args() -> TrainingConfig:
        parser = ArgumentParser()

        group = parser.add_argument_group(title="Persistence Parameters")
        group.add_argument(
            "--save_dir", required=True, help="Where to save the run data"
        )
        group.add_argument("--run_id", help="The id to use for the current run")
        group.add_argument(
            "--data_root",
            required=True,
            help="The directory where the data is stored",
        )
        group.add_argument(
            "--smoke_test",
            help="Whether to run a quick smoke test or not",
            action="store_true",
        )

        group = parser.add_argument_group(title="Training Hyperparameters")
        group.add_argument(
            "--max_epochs", help="the maximum number of epochs to train for"
        )
        group.add_argument("--batch_size", help="The batch size to use")
        group.add_argument("--accumulate_grad_batches", help="The number of batches to process before performing an optimizer step.")
        group.add_argument("--lr", help="The learning rate to use")
        group.add_argument("--lr_decay", help="The learning rate decay to use")
        group.add_argument(
            "--lr_decay_patience",
            help="The number of epochs to wait before decaying the learning rate",
        )
        group.add_argument(
            "--lr_delta_threshold",
            help="The threshold to consider an improvement in the loss for the ReduceLROnPleateau scheduler",
        )
        group.add_argument(
            "--early_stop_patience",
            help="The number of epochs to wait before early stopping",
        )
        group.add_argument(
            "--early_stop_threshold",
            help="The threshold to consider an improvement in the loss for the EarlyStopping callback",
        )
        group.add_argument(
            "--weight_decay",
            help="The weight decay used by AdamW.",
        )

        group = parser.add_argument_group(title="Neural Network Training Parameters")
        group.add_argument(
            "--num_ssms",
            help="The number of independant SSMs to train. Must divide channel_dim",
        )
        group.add_argument(
            "--num_layers",
            help="the number of horizontally stacked SSM blocks",
        )
        group.add_argument(
            "--hidden_dim",
            help="The dimension of the hidden state in each SSM",
        )
        group.add_argument(
            "--channel_dim",
            help="The number of vertically stacked SSM layers",
        )
        group.add_argument(
            "--min_dt",
            help="The minimal dt value at initialization",
        )
        group.add_argument(
            "--max_dt",
            help="The maximal dt value at initialization",
        )
        group.add_argument(
            "--clip_B",
            help="Whether to clip B values at initialization",
        )
        group.add_argument(
            "--kernel_dropout_prob",
            help="Dropout applied to kernel elements",
        )
        group.add_argument(
            "--block_dropout_prob",
            help="Dropout applied after the kernel and layer_activation forward pass",
        )
        group.add_argument(
            "--pre_norm",
            help="Whether to normalize before or after the SSM in a SSM block",
        )
        group.add_argument(
            "--norm",
            help="The normalization layer to use. Can be either layer or batch. Default is layer",
        )
        group.add_argument(
            "--layer_activation",
            help="Activation function applied after kernel forward pass",
        )
        group.add_argument(
            "--final_activation",
            help="Activation applied at the S4 block output",
        )
        group.add_argument("--seq_len", help="The expected sequence length as input")
        return ConfigParser._resolve_args(parser.parse_args())

    @staticmethod
    def _resolve_args(args: Namespace) -> TrainingConfig:
        config = TrainingConfig()
        for arg, value in vars(args).items():
            if value is None:
                continue
            if isinstance(value, str) and ";" in value:
                raise ValueError("Optuna is not supported yet.")
            elif isinstance(value, str) and "," in value:
                raise ValueError("Optuna is not supported yet.")
            else:
                if isinstance(config[arg], str):
                    config[arg] = value
                elif isinstance(config[arg], int):
                    config[arg] = int(value)
                elif isinstance(config[arg], float):
                    config[arg] = float(value)
                elif isinstance(config[arg], bool):
                    config[arg] = bool(value.lower() == "true")
        return config
