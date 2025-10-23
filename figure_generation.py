from pathlib import Path
from typing import List, Union

from matplotlib.pyplot import subplots, show
from numpy import arange
from pandas import DataFrame, concat, read_csv
from seaborn import lineplot, set_palette

set_palette("colorblind")


def plot_learning_curves(
    runs: Union[Path, List[Path]], logger_versions: Union[List[str], str]
) -> None:
    if isinstance(runs, Path):
        runs = [runs]
    if isinstance(logger_versions, str):
        logger_versions = [logger_versions]
    for run, logger_version in zip(runs, logger_versions):
        run_data = _load_data_for_run(run, logger_version)
        train_acc = run_data["train_accuracy"].dropna()
        valid_acc = run_data["valid_accuracy"].dropna()
        train_loss = run_data["train_loss"].dropna()
        valid_loss = run_data["valid_loss"].dropna()

        fig, ax = subplots(nrows=1, ncols=2)
        lineplot(x=arange(len(train_acc)), y=train_acc, label="Training", ax=ax[1])
        lineplot(x=arange(len(valid_acc)), y=valid_acc, label="Validation", ax=ax[1])
        ax[1].set_title("Accuracy Evolution During Training")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].grid(axis="y", alpha=0.4)
        lineplot(x=arange(len(train_loss)), y=train_loss, label="Training", ax=ax[0])
        lineplot(x=arange(len(valid_loss)), y=valid_loss, label="Validation", ax=ax[0])
        ax[0].set_title("Loss Evolution During Training")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].grid(axis="y", alpha=0.4)
        fig.tight_layout()


def _load_data_for_run(run_dir: Path, logger_version: str) -> DataFrame:
    run_dicts = []
    for logger_dir in (run_dir / "logs").rglob(logger_version):
        run_dicts.append(read_csv(logger_dir / "metrics.csv"))
    run_data = concat(run_dicts, axis=0).sort_values("epoch")
    return run_data


if __name__ == "__main__":
    run_loc = Path(
        "/Users/ludovic/Workspace/ssm-speech-processing/training-runs/fir/ssm-speech-processing/google-speech-commands-small/08270690"
    )
    if True:
        plot_learning_curves(run_loc, logger_versions="version_*")
    show()
