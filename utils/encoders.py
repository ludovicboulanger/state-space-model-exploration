from torch import arange, cumsum, Tensor, zeros, zeros_like
from torch.nn import Module
from torch.nn.functional import pad
from torchaudio.transforms import Resample


class PDMEncoder:
    """
    PDM encoding Algorithm : https://github.com/NECOTIS/Keyword-Spotting-with-PDM/blob/main/gsc_dataset.py
    """

    def __init__(self, pdm_factor: int = 10, orig_freq: int = 16000) -> None:
        self.pdm_factor = pdm_factor
        self.upsampler = Resample(orig_freq=orig_freq, new_freq=orig_freq * pdm_factor)
        self.th = 1.0

    def __call__(self, waveform: Tensor) -> Tensor:
        waveform = (waveform / 2) + 0.5
        if self.pdm_factor != 1:
            waveform = self.upsampler(waveform)
        spikes = zeros_like(waveform)
        waveform = waveform.double()
        waveform_cumsum = cumsum(waveform, dim=1)
        waveform_div = waveform_cumsum // self.th
        waveform_div_diff = waveform_div - pad(waveform_div[:, :-1], (1, 0), value=-1)
        spikes[waveform_div_diff > 0] = 1.0
        return spikes

    def forward_sequential(self, waveform) -> Tensor:
        waveform = (waveform / 2) + 0.5
        waveform = self.upsampler(waveform)
        n = waveform.shape[-1]
        y = zeros(size=(n,))
        error = zeros(size=(n + 1,))
        for i in range(n):
            y[i] = 1 if waveform[0, i] >= error[i] else 0
            error[i + 1] = y[i] - waveform[0, i] + error[i]
        return y.unsqueeze(dim=0)


class Upsampler:
    """
    PDM encoding Algorithm : https://github.com/NECOTIS/Keyword-Spotting-with-PDM/blob/main/gsc_dataset.py
    """

    def __init__(self, factor: int = 10, orig_freq: int = 16000) -> None:
        self._factor = factor
        self._upsampler = Resample(orig_freq=orig_freq, new_freq=orig_freq * factor)

    def __call__(self, waveform: Tensor) -> Tensor:
        return self._upsampler(waveform)


class Decimator(Module):
    def __init__(self, fs: int, decimation_factor: int, antialias: bool = True) -> None:
        super().__init__()
        self._decimation_factor = decimation_factor
        self._antialias = antialias
        if antialias:
            self._resample = Resample(orig_freq=fs, new_freq=fs // decimation_factor)

    def update_sequence_params(self, fs: int, decimation_factor: int) -> None:
        self._decimation_factor = decimation_factor
        if self._antialias:
            self._resample = Resample(orig_freq=fs, new_freq=fs // decimation_factor)

    def forward(self, x: Tensor) -> Tensor:
        if self._decimation_factor == 1:
            return x
        if not self._antialias:
            samples = arange(0, x.shape[1], step=self._decimation_factor).long()
            return x[:, samples, :]
        else:
            return self._resample(x.transpose(-1, -2)).transpose(-1, -2)


class LegendreProjection(Module):
    def __init__(self, seq_len: int, decimation_factor: int) -> None:
        super().__init__()
        self._eval_points = arange(-1, -1, seq_len // decimation_factor)

    def forward(self, x: Tensor) -> Tensor:
        return x
