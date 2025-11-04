from pathlib import Path
from typing import List, Union

from matplotlib.pyplot import savefig, subplots, show
from numpy import arange
from pandas import DataFrame, concat, read_csv
from seaborn import lineplot, set_palette, barplot, color_palette

set_palette("colorblind")
PALETTE = color_palette("colorblind")


def plot_classification_learning_curves(runs: Union[Path, List[Path]], logger_versions: Union[List[str], str]) -> None:
    if isinstance(runs, Path):
        runs = [runs]
    if isinstance(logger_versions, str):
        logger_versions = [logger_versions]
    for run, logger_version in zip(runs, logger_versions):
        run_data = _load_data_for_run(run, logger_version).sort_values("epoch")
        train_acc = run_data["train_accuracy_epoch"].dropna()
        valid_acc = run_data["valid_accuracy"].dropna()
        train_loss = run_data["train_loss_epoch"].dropna()
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


def plot_regression_learning_curves(runs: Union[Path, List[Path]], logger_versions: Union[List[str], str]) -> None:
    if isinstance(runs, Path):
        runs = [runs]
    if isinstance(logger_versions, str):
        logger_versions = [logger_versions]
    for run, logger_version in zip(runs, logger_versions):
        run_data = _load_data_for_run(run, logger_version).sort_values("epoch")
        valid_pesq = run_data["valid_pesq"].dropna()
        valid_stoi = run_data["valid_stoi"].dropna()
        valid_loss = run_data["valid_loss"].dropna()
        train_loss = run_data["train_loss_epoch"].dropna()

        fig, ax = subplots(nrows=1, ncols=3)
        lineplot(x=arange(len(train_loss)), y=train_loss, label="Training", ax=ax[0])
        lineplot(x=arange(len(valid_loss)), y=valid_loss, label="Validation", ax=ax[0])
        ax[0].set_title("Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Accuracy")
        ax[0].grid(axis="y", alpha=0.4)
        lineplot(x=arange(len(valid_pesq)), y=valid_pesq, ax=ax[1], color=PALETTE[1])
        ax[1].set_title("PESQ")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("PESQ")
        ax[1].grid(axis="y", alpha=0.4)
        lineplot(x=arange(len(valid_stoi)), y=valid_stoi, ax=ax[2], color=PALETTE[1])
        ax[2].set_title("STOI")
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("STOI")
        ax[2].grid(axis="y", alpha=0.4)
        fig.suptitle("Metrics Evolution During Training")
        fig.tight_layout()
    savefig("learning.png", dpi=400)


def plot_pdm_performance(run_loc: Path) -> None:
    pdm_factor_8_data = _load_data_for_run(run_loc, logger_version="test-8")
    pdm_factor_16_data = _load_data_for_run(run_loc, logger_version="test-16")
    pdm_factor_32_data = _load_data_for_run(run_loc, logger_version="test-32")
    pdm_factor_64_data = _load_data_for_run(run_loc, logger_version="test-64")
    pcm_performance = _load_data_for_run(run_loc, logger_version="version_0")
    pcm_performance = pcm_performance["valid_accuracy"].max()

    fig, ax = subplots()
    bar_heights = [
        100 * pdm_factor_8_data["test_accuracy_step"].mean(),
        100 * pdm_factor_16_data["test_accuracy_step"].mean(),
        100 * pdm_factor_32_data["test_accuracy_step"].mean(),
        100 * pdm_factor_64_data["test_accuracy_step"].mean(),
    ]
    pdm_factors = [8 * 16, 16 * 16, 32 * 16, 64 * 16]
    barplot(x=pdm_factors, y=bar_heights, ax=ax)
    ax.hlines(y=pcm_performance * 100, xmin=-1, xmax=4, linestyles="--", color="k")
    ax.text(x=2, y=pcm_performance * 100 + 2, s="PCM Classification Accuracy")
    ax.set_ylim(0, 100)
    ax.set_xlabel("PDM Sampling Frequency [kHz]")
    ax.set_ylabel("Classification Accuracy [%]")
    ax.set_title("Effect of PDM Upsampling Factor on Classification accuracy at Test Time")


def _load_data_for_run(
    run_dir: Path,
    logger_version: str,
) -> DataFrame:
    run_dicts = []
    for logger_dir in (run_dir / "logs").rglob(logger_version):
        run_dicts.append(read_csv(logger_dir / "metrics.csv"))
    run_data = concat(run_dicts, axis=0)
    return run_data


if __name__ == "__main__":
    run_loc = Path(
        "/home/ludovic/workspace/ssm-speech-processing/training-runs/local/ssm-speech-processing/voicebank_demand/646879"
    )
    if False:
        plot_classification_learning_curves(run_loc, logger_versions="version_*")
    if True:
        plot_regression_learning_curves(run_loc, logger_versions="version_*")
    if False:
        plot_pdm_performance(run_loc)
    # show()
