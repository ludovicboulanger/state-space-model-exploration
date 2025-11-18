from typing import Tuple

from numpy import arange, linspace
from scipy.special import eval_sh_legendre
from torch import Tensor, empty, eye, from_numpy, no_grad, zeros
from torch.fft import rfft, irfft
from torch.nn import Module, Buffer, Linear
from torch.linalg import matrix_exp
from tqdm import tqdm

from utils.registries import StateMatricesRegistry


class SSMEncoder(Module):
    """
    Naively implements an SSM without any optimizations introduced by S4 et alternatives.
    Simply to encode a signal into a memory. In other words, A, B, dt are note made to be
    learnable parameters.
    """

    def __init__(self, mode: str, state_dim: int, output_dim: int, memory_size_seconds: float, seq_len: int) -> None:
        super().__init__()
        self._state_dim = state_dim
        self._theta = memory_size_seconds
        self._seq_len = seq_len
        self._mode = mode
        self._output_dim = output_dim
        self._output_layer = Linear(in_features=self._state_dim, out_features=self._output_dim, bias=False)
        self._init_memory_matrices()

    def recompute_impulse_response(self, upsampling_factor: float) -> None:
        self._seq_len = int(self._seq_len * upsampling_factor)
        self._init_memory_matrices()

    def forward(self, u: Tensor) -> Tensor:
        # Equation (26)
        U = rfft(u, n=2 * self._seq_len, dim=1)
        H = rfft(self._h, n=2 * self._seq_len, dim=0)
        M = U * H.unsqueeze(dim=0)
        m = irfft(M, dim=1)[:, : self._seq_len, :].real
        m = self._output_layer(m)
        return m

    def _init_memory_matrices(self) -> None:
        A, B = StateMatricesRegistry.instantiate(self._mode, self._state_dim)
        if self._mode == "lmu":
            A = A / (1.0 * self._theta)
            B = B / (1.0 * self._theta)
        A_bar, B_bar = self._dicretize_zoh(A, B)
        h = self._compute_impulse_response(A_bar, B_bar)
        self._register_params(A_bar, B_bar, h)

    def _compute_impulse_response(self, A_bar: Tensor, B_bar: Tensor) -> Tensor:
        with no_grad():
            A_i = eye(self._state_dim)
            impulse = zeros(size=(self._state_dim, self._seq_len))
            impulse[:, 0] = B_bar
            for i in tqdm(range(1, self._seq_len), desc="Computing SSM Encoder Impulse Response"):
                A_i = A_i @ A_bar
                impulse[:, i] = A_i @ B_bar
        return impulse.transpose(-1, -2)

    def _dicretize_zoh(self, A: Tensor, B: Tensor) -> Tuple[Tensor, Tensor]:
        """
        https://en.wikipedia.org/wiki/Discretization#discrete_function

        Zero Order Hold discretization
        """
        M = zeros(size=(self._state_dim + 1, self._state_dim + 1))
        M[: self._state_dim, : self._state_dim] = A
        M[: self._state_dim, -1:] = B
        D = matrix_exp(M * 1 / self._seq_len)
        A_bar = D[: self._state_dim, :-1]
        B_bar = D[: self._state_dim, -1]
        return A_bar, B_bar

    def _register_params(self, A_bar: Tensor, B_bar: Tensor, h: Tensor) -> None:
        self._A_bar = Buffer(A_bar)
        self._B_bar = Buffer(B_bar)
        self._h = Buffer(h)

    def _compute_C(self) -> Module:
        layer = Linear(in_features=self._state_dim, out_features=self._output_dim, bias=False)
        if self._mode == "lmu":
            # Project onto the Legendre Polynomials
            eval_points = linspace(start=0, stop=1, num=self._output_dim)
            legendre_degrees = arange(self._state_dim)
            C = empty(size=(self._output_dim, self._state_dim)).float()
            for i in range(self._output_dim):
                eval_point = eval_points[i]
                proj = eval_sh_legendre(legendre_degrees, eval_point)
                C[i, :] = from_numpy(proj).float()
            layer.weight.data = C
            layer.weight.requires_grad = False
        return layer
