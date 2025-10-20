from typing import List, Tuple
from torch import Tensor, clamp_max, complex64, normal, ones, hstack, zeros
from torch.fft import rfft, irfft
from torch.nn import Sigmoid, GELU, Module, Parameter, Linear, LayerNorm, Dropout

from utils import discretize_DPLR, generate_DPLR_HiPPO, kernel_DPLR, compare_tensors


class S4Block(Module):
    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        seq_len: int,
        step: float,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self._prenorm = LayerNorm(normalized_shape=channels)
        self._output_layer = Linear(in_features=channels, out_features=channels)
        self._dropout = Dropout(p=dropout_prob)
        self._non_linearity = GELU()
        self._output_non_linearity = Sigmoid()
        self._ssm = S4Model(
            in_channels=channels,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            step=step,
        )

    @property
    def inference_mode(self) -> bool:
        return self._ssm.inference_mode

    @inference_mode.setter
    def inference_mode(self, mode: bool) -> None:
        self._ssm.inference_mode = mode

    def forward(self, input: Tensor) -> Tensor:
        y = self._prenorm(input)
        y = self._ssm(y)
        y = self._non_linearity(y)
        y = self._dropout(y)
        y = self._output_layer(y)
        return y + input


class S4Model(Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        seq_len: int,
        step: float,
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._hidden_dim = hidden_dim
        self._seq_len = seq_len
        # TODO: Dont learn the step for now
        self._step = step
        self._inference_mode = False
        self._init_DPLR_HiPPO_matrix()

    @property
    def inference_mode(self) -> bool:
        return self._inference_mode

    @inference_mode.setter
    def inference_mode(self, mode: bool) -> None:
        self._inference_mode = mode

    def forward(self, input: Tensor) -> Tensor:
        if self._inference_mode:
            output = self._forward_recurrence(input)
        else:
            output = self._forward_convolution(input)
        return output + self._D.view(-1) * input

    def _forward_convolution(self, input: Tensor) -> Tensor:
        _, time, _ = input.shape
        # TODO: Real Part of Kernel like in the Annotated S4
        kernel = self._generate_kernel().real.unsqueeze(dim=0)
        kernel_fft = rfft(kernel, dim=1, n=2 * time)
        input_fft = rfft(input, dim=1, n=2 * time)
        # TODO: Only return real values
        return irfft(input_fft * kernel_fft, dim=1)[:, :time, :]

    def _forward_recurrence(self, input: Tensor) -> Tensor:
        batch, time, channels = input.shape
        A_bar, B_bar, C_bar = self._discretize()
        u = input.type(A_bar.dtype)
        x_i = zeros(
            size=(batch, channels, self._hidden_dim, 1),
            dtype=A_bar.dtype,
        ).to(u.device)
        outputs: List[Tensor] = []
        A_bar = A_bar.unsqueeze(dim=0)
        B_bar = B_bar.unsqueeze(dim=0)
        C_bar = C_bar.unsqueeze(dim=0)
        for i in range(time):
            input_at_t = u[:, i, :].reshape(batch, channels, 1, 1)
            x_i = A_bar @ x_i
            x_i += B_bar @ input_at_t
            out = C_bar @ x_i
            outputs.append(out.reshape(batch, 1, channels))
        out = hstack(outputs)
        return out.real

    def _init_DPLR_HiPPO_matrix(self) -> None:
        # TODO: For now, we only learn C and D
        Lambda, P, B, _ = generate_DPLR_HiPPO(self._hidden_dim)
        C = normal(mean=0.0, std=0.5**0.5, size=(self._hidden_dim,), dtype=complex64)
        D = ones(size=(1,))

        Lambda = Lambda.unsqueeze(dim=0).repeat(self._in_channels, 1)
        P = P.unsqueeze(dim=0).repeat(self._in_channels, 1)
        B = B.unsqueeze(dim=0).repeat(self._in_channels, 1)
        C = C.unsqueeze(dim=0).repeat(self._in_channels, 1)
        D = D.unsqueeze(dim=0).repeat(self._in_channels, 1)

        self._L_real = Parameter(data=Lambda.real, requires_grad=True)
        self._L_imag = Parameter(data=Lambda.imag, requires_grad=True)
        self._P = Parameter(data=P, requires_grad=True)
        self._B = Parameter(data=B, requires_grad=True)
        self._C = Parameter(data=C, requires_grad=True)
        self._D = Parameter(data=D, requires_grad=True)
        self.register_parameter(name="_L_real", param=self._L_real)
        self.register_parameter(name="_L_imag", param=self._L_imag)
        self.register_parameter(name="_P", param=self._P)
        self.register_parameter(name="_B", param=self._B)
        self.register_parameter(name="_C", param=self._C)
        self.register_parameter(name="_D", param=self._D)

    def _generate_kernel(self) -> Tensor:
        return kernel_DPLR(
            clamp_max(self._L_real, 1e-4) + 1j * self._L_imag,
            self._P,
            self._P,
            self._B,
            self._C,
            self._step,
            self._seq_len,
        )

    def _discretize(self) -> Tuple[Tensor, Tensor, Tensor]:
        return discretize_DPLR(
            clamp_max(self._L_real, 1e-4) + 1j * self._L_imag,
            self._P,
            self._P,
            self._B,
            self._C,
            self._step,
            self._seq_len,
        )


if __name__ == "__main__":
    from torch import rand

    batch = 2
    channels = 4
    time = 16
    hidden_dim = 8

    test_input = rand(size=(batch, time, channels))
    ssm = S4Model(
        in_channels=channels,
        hidden_dim=hidden_dim,
        seq_len=time,
        step=1 / time,
    )
    ssm.inference_mode = False
    conv_output = ssm(test_input)
    ssm.inference_mode = True
    rec_output = ssm(test_input)

    print("Comparing Conv versus Recurrent Outputs for SSM Model")
    compare_tensors(conv_output, rec_output)

    ssm = S4Block(
        channels=channels,
        hidden_dim=hidden_dim,
        seq_len=time,
        step=1 / time,
        dropout_prob=0.0,
    )
    ssm.inference_mode = False
    conv_output = ssm(test_input)
    ssm.inference_mode = True
    rec_output = ssm(test_input)

    print("Comparing Conv versus Recurrent Outputs for SSM Block")
    compare_tensors(conv_output, rec_output)
