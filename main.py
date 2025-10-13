from numpy import hanning
from numpy.fft import fft, fftfreq, fftshift
from torch import Tensor, zeros
from matplotlib.pyplot import figure, show, subplots
from scipy.signal import ShortTimeFFT

from model import SSM
from speech_commands_dataset import SpeechCommandsDataset


def plot_stft(input: Tensor, ssm_output: Tensor) -> None:
    speech = input.detach().numpy().squeeze()
    fig = figure(figsize=(9, 8))
    gs = fig.add_gridspec(nrows=2, ncols=2)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(speech)
    ax0.set_title("Speech Waveform")
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(
        abs(ssm_output.detach().numpy().squeeze()[:64]),
        aspect="auto",
        origin="lower",
        cmap="Blues",
    )
    ax1.set_title("SSM Output")

    ax2 = fig.add_subplot(gs[1, 1])
    stft = ShortTimeFFT(win=hanning(M=128), hop=1, scale_to="magnitude", fs=16000)
    scipy_stft = stft.stft(speech)
    ax2.imshow(
        abs(scipy_stft),
        aspect="auto",
        origin="lower",
        cmap="Blues",
    )
    ax2.set_title("Scipy Output")
    fig.tight_layout()


def plot_frequency_response(ssm: SSM) -> None:
    impulse = zeros(size=(1, 1, 100))
    impulse[..., 0] = 1
    impulse_response = ssm(impulse).squeeze().detach().numpy()
    freq_response = fftshift(fft(impulse_response, n=1024, axis=-1))
    freqs = fftshift(fftfreq(freq_response.shape[-1])) * 16000
    fig, ax = subplots()
    ax.set_title("Frequency Response of the SSM A matrix")
    for i in range(freq_response.shape[0]):
        ax.plot(freqs, abs(freq_response[i]))
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Frequency Response")
    fig.tight_layout()


if __name__ == "__main__":
    dataset = SpeechCommandsDataset(root="./data")
    model = SSM(
        hidden_dim=128,
        step=1 / 16000,
        init="stft",
        init_discrete=True,
        accelerator="cpu",
    )
    model.training = False

    plot_frequency_response(model)

    x, y = dataset[3221]
    x = x.unsqueeze(dim=0)
    output = model(x)
    plot_stft(x, output)
    show()
