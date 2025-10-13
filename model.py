from typing import List, Tuple

from torch import (
    arange,
    complex64,
    exp,
    empty,
    Tensor,
    eye,
    zeros,
    stack,
    ones_like,
    pi,
    real,
)
from torch.fft import rfft, irfft
from torch.linalg import inv, matrix_power
from torch.nn import Module, Parameter
from torch.nn.functional import pad
from torch.nn.init import xavier_normal_


class SSM(Module):
    def __init__(
        self,
        hidden_dim: int,
        step: float = 1.0,
        init: str = "random",
        init_discrete: bool = False,
        requires_grad: bool = True,
        accelerator: str = "cpu",
    ) -> None:
        super().__init__()
        self._accelerator = accelerator
        self._hidden_dim = hidden_dim
        self._step = step
        self._init = init
        self._init_discrete = init_discrete
        self._requires_grad = requires_grad

        if self._init == "random":
            self._init_weights_random()
        elif self._init == "stft":
            self._init_weights_stft()
        else:
            raise ValueError("Unrecognized init value")

        self.register_parameter("_A", param=self._A)
        self.register_parameter("_B", param=self._B)
        self.register_parameter("_C", param=self._C)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return self._forward_pass_convolutional(x)
        else:
            return self._forward_pass_recurrent(x)

    def _forward_pass_recurrent(self, u: Tensor) -> Tensor:
        batch, _, time = u.shape
        A_bar, B_bar, C_bar = self._discretize()
        u = u.type(A_bar.dtype)
        x_i = zeros(
            size=(
                batch,
                self._hidden_dim,
            ),
            device=self._accelerator,
            dtype=A_bar.dtype,
        ).to(u.device)
        outputs: List[Tensor] = []
        for i in range(time):
            x_i = x_i @ A_bar.T + u[..., i] @ B_bar.T
            out = x_i @ C_bar.T
            outputs.append(out)
        return stack(outputs, dim=-1)

    def _forward_pass_convolutional(self, u: Tensor) -> Tensor:
        _, _, time = u.shape
        kernel = self._build_convolution_kernel(time)
        u = pad(u, pad=(u.shape[-1], 0))
        kernel = pad(kernel, pad=(kernel.shape[-1], 0))
        kernel_fft = rfft(kernel, dim=-1)
        input_fft = rfft(u, dim=-1)
        return real(irfft(input_fft * kernel_fft, dim=-1))[..., :time]

    def _init_weights_random(self) -> None:
        self._A = Parameter(
            data=empty(
                size=(self._hidden_dim, self._hidden_dim), device=self._accelerator
            ),
            requires_grad=self._requires_grad,
        )
        self._B = Parameter(
            data=empty(size=(self._hidden_dim, 1), device=self._accelerator),
            requires_grad=self._requires_grad,
        )
        self._C = Parameter(
            data=empty(size=(1, self._hidden_dim), device=self._accelerator),
            requires_grad=self._requires_grad,
        )
        xavier_normal_(self._A.data)
        xavier_normal_(self._B.data)
        xavier_normal_(self._C.data)

    def _init_weights_stft(self) -> None:
        self._A = Parameter(
            data=empty(
                size=(self._hidden_dim, self._hidden_dim),
                device=self._accelerator,
                dtype=complex64,
            ),
            requires_grad=self._requires_grad,
        )
        self._B = Parameter(
            data=empty(
                size=(self._hidden_dim, 1), device=self._accelerator, dtype=complex64
            ),
            requires_grad=self._requires_grad,
        )
        self._C = Parameter(
            data=empty(
                size=(self._hidden_dim, self._hidden_dim),
                device=self._accelerator,
                dtype=complex64,
            ),
            requires_grad=self._requires_grad,
        )

        self._B.data = ones_like(self._B.data)
        damp = 0.99
        frequency_steps = arange(start=0, end=self._hidden_dim, step=1)
        coefficients = damp * exp(-1j * 2 * pi * frequency_steps / self._hidden_dim)
        self._A.data = eye(self._hidden_dim) * coefficients.view(-1, 1)
        self._C.data = eye(self._hidden_dim, dtype=complex64)

    def _build_convolution_kernel(self, length: int) -> Tensor:
        A_bar, B_bar, C_bar = self._discretize()
        kernel_elements = []
        for i in range(length):
            kernel_elements.append(C_bar @ matrix_power(A_bar, i) @ B_bar)
        kernel = stack(kernel_elements).squeeze()
        return kernel

    def _discretize(self) -> Tuple[Tensor, Tensor, Tensor]:
        identity = eye(n=self._hidden_dim)
        tmp = inv(identity - self._step / 2 * self._A)
        if self._init_discrete:
            A_bar = self._A
            B_bar = self._B
            C_bar = self._C
        else:
            A_bar = tmp @ (identity + self._step / 2 * self._A)
            B_bar = tmp @ (self._step * self._B)
            C_bar = self._C
        return A_bar, B_bar, C_bar
