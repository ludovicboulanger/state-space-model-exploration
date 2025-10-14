from numpy import hanning
from numpy.fft import fft, fftfreq, fftshift
from torch import Tensor, arange, complex64, exp, ones, pi, zeros, rand
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from matplotlib.pyplot import figure, show, subplots
from scipy.signal import ShortTimeFFT

from model import SSMLayer, SSMNetwork
from speech_commands_dataset import SpeechCommandsDataset


def plot_stft(input: Tensor, ssm_output: Tensor, title: str) -> None:
    freqs = ssm_output.shape[0]
    speech = input.detach().numpy().squeeze()
    fig = figure(figsize=(9, 8))
    gs = fig.add_gridspec(nrows=2, ncols=2)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(speech)
    ax0.set_title("Speech Waveform")
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(
        abs(ssm_output.detach().numpy().squeeze()[: freqs // 2]),
        aspect="auto",
        origin="lower",
        cmap="Blues",
    )
    ax1.set_title(title)

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


def plot_frequency_response(ssm: SSMLayer) -> None:
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


def run_ssm_as_stft() -> None:
    fft_size = 64
    dataset = SpeechCommandsDataset(root="./data")

    damp = 0.99
    frequency_steps = arange(start=0, end=fft_size, step=1)
    coefficients = damp * exp(-1j * 2 * pi * frequency_steps / fft_size)
    A = coefficients.view(-1, 1, 1)
    B = ones(size=(fft_size, 1, 1), dtype=complex64)
    C = ones(size=(fft_size, 1, 1), dtype=complex64)

    model = SSMLayer(
        hidden_dim=1,
        num_features=64,
        step=1 / 16000,
        A_init=A,
        B_init=B,
        C_init=C,
        init_discrete=True,
        accelerator="cpu",
    )
    x, _ = dataset[3221]
    x = x.unsqueeze(dim=0)
    model.training = False
    rec_output = model(x)
    plot_stft(x, rec_output.squeeze(), title="SSM Output (Recurrence)")
    model.training = True
    conv_output = model(x)
    plot_stft(x, conv_output.squeeze(), title="SSM Output (Convolution)")
    max_error = (rec_output - conv_output).abs().max()
    relative_error = max_error / rec_output.abs().max()
    print(f"Max error: {max_error}, Relative: {relative_error}")


def compare_conv_versus_recurrent_view() -> None:
    model = SSMLayer(
        hidden_dim=1,
        num_features=4,
        step=1 / 16000,
        accelerator="cpu",
    )

    model.training = True
    x = rand(size=(4, 1, 5000))
    from torch import real

    conv_output = real(model(x))
    print(conv_output)
    model.training = False
    rec_output = model(x)
    print(rec_output)

    max_error = (rec_output - conv_output).abs().max()
    relative_error = max_error / rec_output.abs().max()
    print(f"Max error: {max_error}, Relative: {relative_error}")


if __name__ == "__main__":
    # compare_conv_versus_recurrent_view()
    # run_ssm_as_stft()
    train_ssm_on_speech_commands()
    show()
