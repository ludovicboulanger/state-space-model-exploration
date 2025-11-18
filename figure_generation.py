from pathlib import Path
from typing import List, Optional, Union

from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.pyplot import savefig, subplots, show, figure
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from numpy import arange, exp, hanning, linspace, zeros
from pandas import DataFrame, concat, read_csv
from scipy.signal import ShortTimeFFT
from seaborn import lineplot, set_palette, barplot, color_palette, set_theme
from soundfile import read
from torch import from_numpy

from config import TrainingConfig
from utils.encoders import PDMEncoder

set_palette("colorblind")
set_theme(style="whitegrid")
PALETTE = color_palette("colorblind")


def plot_audio_signal() -> None:
    audio_loc = Path(__file__).parent / "data/SpeechCommands/speech_commands_v0.02/backward/0a2b400e_nohash_0.wav"
    audio, fs = read(audio_loc)
    fig, ax = subplots()
    lineplot(audio)
    ax.set_xlabel("Time [samples]")
    ax.set_ylabel("Amplitude")
    ax.set_title('Speech Signal for "Backward"')


def plot_audio_vs_pdm_spectrograms() -> None:
    audio_loc = Path(__file__).parent / "data/SpeechCommands/speech_commands_v0.02/backward/0a2b400e_nohash_0.wav"
    audio, fs = read(audio_loc)
    time_pcm = arange(len(audio)) / fs

    audio_tensor = from_numpy(audio).float().view(1, -1)
    pdm_audio = PDMEncoder(pdm_factor=64, orig_freq=fs)(audio_tensor)
    pdm_audio = pdm_audio.squeeze().numpy()
    pdm_audio = 2 * pdm_audio - 1

    time_pdm = arange(len(pdm_audio)) / (64 * fs)

    fig = figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 2], hspace=0.7)
    ax_top_left = fig.add_subplot(gs[0, 0])
    ax_top_right = fig.add_subplot(gs[0, 1])
    ax_bottom_left = fig.add_subplot(gs[1, 0])
    ax_bottom_right = fig.add_subplot(gs[1, 1])

    lineplot(x=time_pcm, y=audio, ax=ax_top_left)
    ax_top_left.set_title("PCM Encoded Data")
    ax_top_left.set_xlabel("Time (s)")

    lineplot(x=time_pdm, y=pdm_audio, ax=ax_top_right)
    ax_top_right.set_title("PDM Encoded Data")

    # inset Axes....
    zoom_start = 0.5
    zoom_end = 0.50005
    pdm_min = -1
    pdm_max = 1

    w, h = 0.7, 0.4
    axins = ax_top_right.inset_axes(
        [(1 - w) / 2, -0.6, w, h],  # type: ignore
        xlim=(zoom_start, zoom_end),
        ylim=(pdm_min, pdm_max),
        yticklabels=[],
    )
    axins.plot(time_pdm, pdm_audio)
    axins.set_ylim(-1.1, 1.1)
    nticks = 6
    ticks_seconds = linspace(zoom_start, zoom_end, nticks)
    tick_labels_us = linspace(0, 50, nticks).astype(int)
    axins.set_xticks(ticks_seconds)
    axins.set_xticklabels(tick_labels_us)
    axins.set_xlabel("Time (µs + 5e5)")

    mark_inset(ax_top_right, axins, loc1=2, loc2=1, fc="none", ec="black", lw=1)
    mark_inset(ax_top_right, axins, loc1=1, loc2=2, fc="none", ec="black", lw=1)

    rect = Rectangle(
        (zoom_start, pdm_min),
        zoom_end - zoom_start,
        pdm_max - pdm_min,
        edgecolor="k",
        facecolor="none",
        lw=1.5,
        zorder=5,
    )
    ax_top_right.add_patch(rect)

    stft = ShortTimeFFT(win=hanning(128), hop=32, fs=fs)
    pcm_spectrogram = stft.spectrogram(audio)
    x_vals = arange(pcm_spectrogram.shape[-1])
    y_vals = arange(pcm_spectrogram.shape[0])[1:]

    ax_bottom_left.pcolormesh(x_vals, y_vals, pcm_spectrogram[1:], shading="gouraud", norm=LogNorm())

    stft = ShortTimeFFT(win=hanning(64 * 128), hop=64 * 32, fs=64 * fs)
    pdm_spectrogram = stft.spectrogram(pdm_audio)
    x_vals = arange(pdm_spectrogram.shape[-1])
    y_vals = arange(pdm_spectrogram.shape[0])[1:]
    ax_bottom_right.pcolormesh(x_vals, y_vals, pdm_spectrogram[1:], shading="gouraud", norm=LogNorm())
    ax_bottom_right.set_yscale("log")

    fig.tight_layout()


def plot_decimated_pcm_results(run_locs: Path) -> None:
    fig, ax0 = subplots(figsize=(12, 8))

    colors = {"lmu": PALETTE[0], "s4": PALETTE[1], "dense": PALETTE[2]}
    lmu_runs = run_locs / "lmu"
    x = []
    y = []
    for run in lmu_runs.iterdir():
        try:
            run_result = _load_data_for_run(run, logger_version="version_*")["valid_accuracy"].max()
        except Exception:
            continue
        decimation = TrainingConfig.from_json(run / "training_config.json")["decimation_factor"]
        x.append(decimation)
        y.append(run_result)
    lineplot(x=x, y=y, marker="D", color=colors["lmu"], ax=ax0, label="LMU")

    s4_runs = run_locs / "s4"
    x = []
    y = []
    for run in s4_runs.iterdir():
        try:
            run_result = _load_data_for_run(run, logger_version="version_*")["valid_accuracy"].max()
        except Exception:
            continue
        decimation = TrainingConfig.from_json(run / "training_config.json")["decimation_factor"]
        x.append(decimation)
        y.append(run_result)
    lineplot(x=x, y=y, marker="D", color=colors["s4"], ax=ax0, label="S4-LegS")

    dense_runs = run_locs / "dense"
    x = []
    y = []
    for run in dense_runs.iterdir():
        try:
            run_result = _load_data_for_run(run, logger_version="version_*")["valid_accuracy"].max()
        except Exception:
            continue
        decimation = TrainingConfig.from_json(run / "training_config.json")["decimation_factor"]
        x.append(decimation)
        y.append(run_result)
    lineplot(x=x, y=y, marker="D", color=colors["dense"], ax=ax0, label="Dense")

    ax0.legend()
    ax0.set_ylim(0, 1)
    ax0.set_xlabel("Decimation Factor")
    ax0.set_ylabel("Classification Accuracy")
    ax0.set_title("Decimated PCM Classfication Accuracy")


def plot_classification_learning_curves(
    runs: Union[Path, List[Path]], logger_versions: Union[List[str], str], title: Optional[str] = None
) -> None:
    if isinstance(runs, Path):
        runs = [runs]
    if isinstance(logger_versions, str):
        logger_versions = [logger_versions]
    for run, logger_version in zip(runs, logger_versions):
        run_data = _load_data_for_run(run, logger_version).sort_values("epoch")
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
        if title is not None:
            fig.suptitle(title)
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


def plot_pdm_performance_per_encoder_and_decimation(run_dir: Path) -> None:
    """
    dense_data = _load_run_for_decimation(run_dir / "dense", logger_versions="valid-pdm-*", decimation=1)
    dense_data["encoder"] = "dense"
    dense_data = dense_data.rename(columns={"test_accuracy": "test_accuracy_epoch"})
    data.append(dense_data)
    """
    fig, ax = subplots(nrows=1, ncols=3, figsize=(10, 4))
    for i, decimation in enumerate([1, 32, 128]):
        data = []
        lmu_data = _load_run_for_decimation(run_dir / "lmu", logger_versions="valid-pdm-*", decimation=decimation)
        lmu_data["encoder"] = "lmu"
        data.append(lmu_data)
        s4_data = _load_run_for_decimation(run_dir / "s4", logger_versions="valid-pdm-*", decimation=decimation)
        s4_data["encoder"] = "s4"
        data.append(s4_data)
        df = concat(data, axis=0)[["logger_version", "test_accuracy_epoch", "encoder"]]
        df["pdm_factor"] = df.apply(lambda row: int(row["logger_version"].split("-")[-1]), axis=1)
        df = df.sort_values(by="pdm_factor", ascending=True)
        df = df.dropna()

        barplot(df, x="pdm_factor", y="test_accuracy_epoch", hue="encoder", ax=ax[i % 3], legend=i == 0)
        ax[i % 3].set_title(f"Decimation of {decimation}")
        ax[i % 3].set_xlabel("")
        ax[i % 3].set_ylabel("")
        if i == 0:
            ax[i % 3].set_ylabel("Validation Accuracy (%)")
            ax[i % 3].legend(loc="lower center")
            ax[i % 3].set_xlabel("PDM Upsampling Factor")
    fig.suptitle("PDM Classification Accuracy")
    fig.tight_layout()


def plot_classification_performance_per_decimation(root: Path) -> None:
    all_data = []
    for run_dir in root.iterdir():
        data = _load_data_for_run(run_dir, "version_0")
        config = TrainingConfig.from_json(run_dir / "training_config.json")
        data["decimation_factor"] = config.decimation_factor
        all_data.append(data)
    df = concat(all_data, axis=0).reset_index(drop=True)
    df = df.groupby(by="decimation_factor").max().reset_index()
    print(df.columns)

    fig, ax = subplots()
    barplot(x="decimation_factor", y="valid_accuracy", data=df)


def plot_pdm_performance(run_loc: Path) -> None:
    x = []
    y = []
    for pdm_factor in [8, 16, 32, 64]:
        try:
            data = _load_data_for_run(run_loc, logger_version=f"valid-pdm-{pdm_factor}")
            x.append(pdm_factor * 16)
            y.append(100 * data["test_accuracy"].mean())
        except Exception:
            print(f"NOTE: No data for PDM factor {pdm_factor}")

    pcm_performance = _load_data_for_run(run_loc, logger_version="version_0")
    pcm_performance = pcm_performance["valid_accuracy"].max()

    fig, ax = subplots()
    barplot(x=x, y=y, ax=ax)
    ax.hlines(y=pcm_performance * 100, xmin=-1, xmax=4, linestyles="--", color="k")
    ax.set_ylim(0, 100)
    ax.set_xlabel("PDM Sampling Frequency [kHz]")
    ax.set_ylabel("Classification Accuracy [%]")
    ax.set_title("Effect of PDM Upsampling Factor on Classification accuracy at Test Time")


def plot_resulting_encoder_dts(run_loc: Path) -> None:
    from torch import load

    data = load(run_loc / "last_checkpoint.ckpt")
    print(data["state_dict"]["_encoder.0._kernel._dt"])


def _load_run_for_decimation(root: Path, logger_versions: str, decimation: int) -> DataFrame:
    for run in (root).iterdir():
        config = TrainingConfig.from_json(run / "training_config.json")
        if config.decimation_factor == decimation:
            data = _load_data_for_run(run, logger_versions)
    return data


def _load_all_runs_from_dir(run_dir: Path, logger_versions: str, attrs: Optional[List] = None) -> DataFrame:
    run_dicts = []
    for folder in run_dir.iterdir():
        data = _load_data_for_run(folder, logger_versions)
        config = TrainingConfig.from_json(run_dir / "training_config.json")
        if attrs is not None:
            for attr in attrs:
                data[attr] = config[attr]
        run_dicts.append(data)
    return concat(run_dicts, axis=0)


def _load_data_for_run(
    run_dir: Path,
    logger_version: str,
) -> DataFrame:
    run_dicts = []
    for logger_dir in (run_dir / "logs").rglob(logger_version):
        df = read_csv(logger_dir / "metrics.csv")
        df["logger_version"] = logger_dir.name
        run_dicts.append(df)
    run_data = concat(run_dicts, axis=0)
    return run_data


if __name__ == "__main__":
    if False:
        plot_classification_learning_curves(
            Path("/home/ludovic/workspace/ssm-speech-processing/training-runs/narval/52104503/s4/52104503_7"),
            logger_versions="version_*",
        )
    if False:
        plot_regression_learning_curves(
            "/home/ludovic/workspace/ssm-speech-processing/training-runs/local/ssm-speech-processing/voicebank_demand/153877",
            "version_*",
        )
    if False:
        plot_pdm_performance("/home/ludovic/workspace/ssm-speech-processing/training-runs/narval/52424530")
    if False:
        plot_decimated_pcm_results(Path("/home/ludovic/workspace/ssm-speech-processing/training-runs/narval/52424530"))
    if True:
        plot_pdm_performance_per_encoder_and_decimation(
            Path("/home/ludovic/workspace/ssm-speech-processing/training-runs/narval/52424530")
        )
    if False:
        plot_classification_performance_per_decimation(
            Path("/home/ludovic/workspace/ssm-speech-processing/training-runs/narval/52424530")
        )
    if False:
        plot_resulting_encoder_dts(
            Path("/home/ludovic/workspace/ssm-speech-processing/training-runs/narval/52104503/s4/52110610_1")
        )

    show()
