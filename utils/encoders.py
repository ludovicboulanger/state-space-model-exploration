from torch import cumsum, Tensor, zeros, zeros_like
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
