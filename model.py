from typing import List, Optional, Tuple

from torch import (
    empty,
    Tensor,
    eye,
    ones,
    zeros,
    stack,
    hstack,
)
from torch.fft import fft, ifft
from torch.linalg import inv, matrix_power
from torch.nn import Module, Parameter
from torch.nn.functional import pad
from torch.nn.init import xavier_normal_


class SSMLayer(Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        step: float = 1.0,
        A_init: Optional[Tensor] = None,
        B_init: Optional[Tensor] = None,
        C_init: Optional[Tensor] = None,
        init_discrete: bool = False,
        requires_grad: bool = True,
        accelerator: str = "cpu",
    ) -> None:
        super().__init__()
        self._accelerator = accelerator
        self._num_features = num_features
        self._hidden_dim = hidden_dim
        self._step = step
        self._init_discrete = init_discrete
        self._requires_grad = requires_grad
        self._init_weights_random()
        if A_init is not None:
            self._A.data = A_init
        if B_init is not None:
            self._B.data = B_init
        if C_init is not None:
            self._C.data = C_init

        self.register_parameter("_A", param=self._A)
        self.register_parameter("_B", param=self._B)
        self.register_parameter("_C", param=self._C)
        self.register_parameter("_D", param=self._D)

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
            size=(batch, self._num_features, self._hidden_dim, 1),
            device=self._accelerator,
            dtype=A_bar.dtype,
        ).to(u.device)
        outputs: List[Tensor] = []
        A_bar = A_bar.unsqueeze(dim=0)
        B_bar = B_bar.unsqueeze(dim=0)
        C_bar = C_bar.unsqueeze(dim=0)
        for i in range(time):
            x_i = A_bar @ x_i
            x_i = x_i + B_bar @ u[..., i : i + 1].unsqueeze(dim=-1)
            out = C_bar @ x_i
            outputs.append(out.squeeze())
        return stack(outputs, dim=-1)

    def _forward_pass_convolutional(self, u: Tensor) -> Tensor:
        _, _, time = u.shape
        u = pad(u, pad=(0, u.shape[-1]))
        kernel = self._build_convolution_kernel(time)
        kernel = pad(kernel, pad=(0, kernel.shape[-1]))
        kernel_fft = fft(kernel, dim=-1)
        input_fft = fft(u, dim=-1)
        return ifft(input_fft * kernel_fft, dim=-1)[..., :time]

    def _init_weights_random(self) -> None:
        self._A = Parameter(
            data=empty(
                size=(self._num_features, self._hidden_dim, self._hidden_dim),
                device=self._accelerator,
            ),
            requires_grad=self._requires_grad,
        )
        self._B = Parameter(
            data=empty(
                size=(self._num_features, self._hidden_dim, 1),
                device=self._accelerator,
            ),
            requires_grad=self._requires_grad,
        )
        self._C = Parameter(
            data=empty(
                size=(self._num_features, 1, self._hidden_dim),
                device=self._accelerator,
            ),
            requires_grad=self._requires_grad,
        )
        self._D = Parameter(
            data=ones(
                size=(1, 1),
                device=self._accelerator,
            ),
            requires_grad=self._requires_grad,
        )
        xavier_normal_(self._A.data)
        xavier_normal_(self._B.data)
        xavier_normal_(self._C.data)

    def _build_convolution_kernel(self, length: int) -> Tensor:
        A_bar, B_bar, C_bar = self._discretize()
        kernel_elements = []
        for i in range(length):
            element = C_bar @ matrix_power(A_bar, i) @ B_bar
            kernel_elements.append(element.view(self._num_features, 1))
        kernel = hstack(kernel_elements)
        return kernel

    def _discretize(self) -> Tuple[Tensor, Tensor, Tensor]:
        identity = eye(n=self._hidden_dim).unsqueeze(dim=0)
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
