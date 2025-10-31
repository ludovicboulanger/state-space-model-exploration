from torch import cumsum, Tensor, diff, view_as_real, where, zeros, zeros_like
from torch.cuda import is_available as is_cuda_available
from torch.nn import (
    Module,
    GELU,
    Sigmoid,
    ReLU,
    GLU,
    Tanh,
    Identity,
    LayerNorm,
    BatchNorm1d,
    GroupNorm
)
from torch.nn.functional import pad
from torchaudio.transforms import Resample


try:
    from extensions.kernels.cauchy import cauchy_mult
    print("Using CUDA Kernels")
except Exception as e:
    print("Error Loading CUDA kernels, {e}\n Falling back to pyekops implementation.")
    from pykeops.torch import Genred
    def cauchy_mult( v: Tensor, z: Tensor, w: Tensor) -> Tensor:
        def _broadcast_dims(*tensors):
            max_dim = max([len(tensor.shape) for tensor in tensors])
            tensors = [
                tensor.view((1,) * (max_dim - len(tensor.shape)) + tensor.shape)
                for tensor in tensors
            ]
            return tensors

        expr_num = "z * ComplexReal(v) - Real2Complex(Sum(v * w))"
        expr_denom = "ComplexMult(z-w, z-Conj(w))"

        cauchy_mult = Genred(
            f"ComplexDivide({expr_num}, {expr_denom})",
            [
                "v = Vj(2)",
                "z = Vi(2)",
                "w = Vj(2)",
            ],
            reduction_op="Sum",
            axis=1,
        )

        v, z, w = _broadcast_dims(v, z, w)
        v = view_as_real(v)
        z = view_as_real(z)
        w = view_as_real(w)

        r = 2 * cauchy_mult(v, z, w, backend="GPU" if is_cuda_available() else "CPU")
        return view_as_complex(r)  # type: ignore


class ActivationRegistry:
    @staticmethod
    def instantiate(activation: str) -> Module:
        if activation == "gelu":
            return GELU()
        elif activation == "glu":
            return GLU()
        elif activation == "sigmoid":
            return Sigmoid()
        elif activation == "relu":
            return ReLU()
        elif activation == "tanh":
            return Tanh()
        elif activation == "none":
            return Identity()
        else:
            raise Exception(f"Invalid activation function provided : {activation}")


class NormRegistry:
    @staticmethod
    def instantiate(norm: str, channels: int) -> Module:
        if norm == "layer":
            return LayerNorm(normalized_shape=channels)
        elif norm == "batch":
            return BatchNorm1d(num_features=channels)
        elif norm == "group":
            return GroupNorm(num_groups=channels, num_channels=channels)
        elif norm == "none":
            return Identity()
        else:
            raise Exception(f"Invalid normalization function provided : {norm}")


class PDMEncoder:
    """
    PDM encoding Algorithm : https://github.com/NECOTIS/Keyword-Spotting-with-PDM/blob/main/gsc_dataset.py
    """
    def __init__(self, pdm_factor: int = 10, orig_freq: int = 16000) -> None:
        self.pdm_factor = pdm_factor
        self.upsampler = Resample(orig_freq=orig_freq, new_freq=orig_freq*pdm_factor)
        self.th = 1.
    
    def __call__(self, waveform: Tensor) -> Tensor:
        waveform = (waveform/2)+0.5
        if self.pdm_factor !=1: waveform = self.upsampler(waveform)
        spikes = zeros_like(waveform)
        waveform = waveform.double()
        waveform_cumsum = cumsum(waveform, dim=1)
        waveform_div = waveform_cumsum//self.th
        waveform_div_diff = waveform_div - pad(waveform_div[:,:-1], (1,0), value=-1)
        spikes[waveform_div_diff>0] = 1.
        return spikes

    def forward_sequential(self, waveform) -> Tensor:
        waveform = (waveform/2)+0.5
        waveform = self.upsampler(waveform)
        n = waveform.shape[-1]
        y = zeros(size=(n,))
        error = zeros(size=(n+1,))    
        for i in range(n):
            y[i] = 1 if waveform[0,i] >= error[i] else 0
            error[i+1] = y[i] - waveform[0,i] + error[i]
        return y.unsqueeze(dim=0) 


class Upsampler:
    """
    PDM encoding Algorithm : https://github.com/NECOTIS/Keyword-Spotting-with-PDM/blob/main/gsc_dataset.py
    """
    def __init__(self, factor: int = 10, orig_freq: int = 16000) -> None:
        self._factor = factor
        self._upsampler = Resample(orig_freq=orig_freq, new_freq=orig_freq*factor)
    
    def __call__(self, waveform: Tensor) -> Tensor:
        return self._upsampler(waveform)