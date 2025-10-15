from typing import Callable, List, Tuple
from torch import (
    Tensor,
    arange,
    complex64,
    empty,
    exp,
    eye,
    diag,
    float32,
    hstack,
    manual_seed,
    matrix_power,
    mean,
    ones_like,
    pi,
    rand,
    sqrt,
    stack,
    tril,
    zeros,
)
from torch.fft import ifft, fft
from torch.nn.init import uniform_, normal_
from torch.linalg import inv, eigh


manual_seed(3221)


######## Standard SSM Functions in order to compare S4 to Standard SSM ##########
def compare_tensors(a: Tensor, b: Tensor) -> None:
    max_error = (a - b).abs().max()
    relative_error = max_error / a.abs().max()
    print(f"Max error: {max_error}, Relative: {relative_error}")


def discretize(
    A: Tensor, B: Tensor, C: Tensor, step: float
) -> Tuple[Tensor, Tensor, Tensor]:
    identity = eye(n=A.shape[0])
    tmp = inv(identity - step / 2 * A)

    A_bar = tmp @ (identity + step / 2 * A)
    B_bar = tmp @ (step * B)
    C_bar = C

    return A_bar, B_bar, C_bar


def K_conv(A_bar: Tensor, B_bar: Tensor, C_bar: Tensor, L: int) -> Tensor:
    kernel_elements = []
    for i in range(L):
        element = C_bar @ matrix_power(A_bar, i) @ B_bar
        kernel_elements.append(element)
    kernel = hstack(kernel_elements)
    return kernel.real


def random_SSM(N, dtype=float32):
    A = empty(size=(N, N), dtype=dtype)
    B = empty(size=(N, 1), dtype=dtype)
    C = empty(size=(1, N), dtype=dtype)
    uniform_(A)
    uniform_(B)
    uniform_(C)
    return A, B, C


def forward_pass_convolution(input: Tensor, kernel: Tensor) -> Tensor:
    _, _, time = input.shape
    kernel_fft = fft(kernel, dim=-1, n=2 * time)
    input_fft = fft(input, dim=-1, n=2 * time)
    return ifft(input_fft * kernel_fft.unsqueeze(dim=0), dim=-1)[..., :time]


def forward_pass_recurrent(
    input: Tensor, A_bar: Tensor, B_bar: Tensor, C_bar: Tensor
) -> Tensor:
    batch, _, time = input.shape
    u = input.type(A_bar.dtype)
    x_i = zeros(
        size=(batch, A_bar.shape[0], 1),
        dtype=A_bar.dtype,
    ).to(u.device)
    A_bar = A_bar.unsqueeze(dim=0)
    B_bar = B_bar.unsqueeze(dim=0)
    C_bar = C_bar.unsqueeze(dim=0)
    outputs: List[Tensor] = []
    for i in range(time):
        x_i = A_bar @ x_i
        x_i = x_i + B_bar @ u[..., i : i + 1]
        out = C_bar @ x_i
        outputs.append(out)
    out = stack(outputs, dim=-1).view(batch, 1, -1)
    return out.real


######## Standard SSM Functions in order to compare S4 to Standard SSM ##########


def build_A_from_DPLR(Lambda: Tensor, P: Tensor, Q: Tensor) -> Tensor:
    return diag(Lambda) - P.view(-1, 1) @ Q.view(-1, 1).conj().T


def generate_hippo_matrix(N: int) -> Tensor:
    P = sqrt(1 + 2 * arange(N))
    A = P.view(-1, 1) * P.view(1, -1)
    A = tril(A) - diag(arange(N))
    return -A


def generate_random_DPLR(N) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    Lambda = empty(size=(N,))
    P = empty(size=(N,))
    Q = empty(size=(N,))
    B = empty(size=(N, 1))
    C = empty(size=(1, N))

    uniform_(Lambda)
    uniform_(P)
    uniform_(Q)
    uniform_(B)
    uniform_(C)
    return Lambda, P, Q, B, C


def K_gen_inverse(
    A_bar: Tensor, B_bar: Tensor, C_bar: Tensor, L: int
) -> Callable[[Tensor], Tensor]:
    I_matrix = eye(A_bar.shape[0]).type(complex64)
    A_bar_L = matrix_power(A_bar, L).type(complex64)
    Ct = C_bar.type(complex64) @ (I_matrix - A_bar_L)

    return lambda z: (
        Ct.conj()
        @ inv(I_matrix - A_bar.type(complex64) * z.view(-1, 1, 1))
        @ B_bar.type(complex64)
    )


def conv_from_gen(gen, L):
    omega_L = exp((-2j * pi) * (arange(L) / L))
    at_roots = gen(omega_L)
    out = ifft(at_roots, n=L, dim=0).reshape(L)
    return out.real


def kernel_DPLR(
    Lambda: Tensor, P: Tensor, Q: Tensor, B: Tensor, C: Tensor, step: float, L: int
) -> Tensor:
    omega_L = exp((-2j * pi) * (arange(L) / L))
    a_term = (C.conj(), Q.conj())
    b_term = (B, P)

    g = (2.0 / step) * ((1.0 - omega_L) / (1.0 + omega_L))
    c = 2.0 / (1.0 + omega_L)

    k00 = cauchy_kernel(a_term[0] * b_term[0].squeeze(), g, Lambda)
    k01 = cauchy_kernel(a_term[0] * b_term[1], g, Lambda)
    k10 = cauchy_kernel(a_term[1] * b_term[0].squeeze(), g, Lambda)
    k11 = cauchy_kernel(a_term[1] * b_term[1], g, Lambda)
    at_roots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    output = ifft(at_roots, dim=0, n=L).reshape(L)
    return output.real


def cauchy_kernel(v: Tensor, omega: Tensor, lambd: Tensor) -> Tensor:
    omega = omega.view(-1, 1)
    return (v / (omega - lambd)).sum(dim=-1)


def discretize_DPLR(
    Lambda: Tensor, P: Tensor, Q: Tensor, B: Tensor, C: Tensor, step: float, L: int
) -> Tuple[Tensor, Tensor, Tensor]:
    B = B.view(-1, 1)
    C = C.view(1, -1)

    N = Lambda.shape[0]
    A = diag(Lambda) - P.view(-1, 1) @ Q.view(-1, 1).conj().T
    I_matrix = eye(N)

    A0 = (2.0 / step) * I_matrix + A

    D = diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)
    A_bar = A1 @ A0
    B_bar = 2 * A1 @ B

    C_bar = C @ inv(I_matrix - matrix_power(A_bar, L)).conj()
    return A_bar, B_bar, C_bar.conj()


def generate_DPLR_HiPPO(N) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    A, P, B = generate_NPLR_HiPPO(N)
    S = A + P.view(-1, 1) * P.view(1, -1)
    S_diag = diag(S)
    Lambda_real = mean(S_diag) * ones_like(S_diag)
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P.type(complex64)
    B = V.conj().T @ B.type(complex64)
    return Lambda_real + 1j * Lambda_imag, P, B, V


def generate_NPLR_HiPPO(N: int) -> Tuple[Tensor, Tensor, Tensor]:
    nhippo = generate_hippo_matrix(N)

    P = sqrt(arange(N) + 0.5)
    B = sqrt(2 * arange(N) + 1.0)

    return nhippo, P, B


##### Testing Functions ######
def test_gen_inverse(L: int = 16, hidden_dim: int = 4) -> None:
    A, B, C = random_SSM(hidden_dim)
    A_bar, B_bar, C_bar = discretize(A, B, C, 1 / L)
    b = K_conv(A_bar, B_bar, C_bar, L=L)

    a = conv_from_gen(K_gen_inverse(A_bar, B_bar, C_bar, L=L), L=L)
    compare_tensors(a, b)


def test_cauchy_kernel(L: int = 16, hidden_dim: int = 4) -> None:
    Lambda, P, Q, B, C = generate_random_DPLR(hidden_dim)
    A = build_A_from_DPLR(Lambda, P, Q)

    A_bar, B_bar, C_bar = discretize(A, B, C, 1.0 / L)
    a = K_conv(A_bar, B_bar, C_bar.conj(), L=L)

    C = (eye(hidden_dim) - matrix_power(A_bar, L)).conj().T @ C_bar.ravel()
    b = kernel_DPLR(Lambda, P, Q, B, C, step=1.0 / L, L=L)

    compare_tensors(a, b)


def test_NPLR_versus_DPLR(N: int = 8) -> None:
    A2, P, B = generate_NPLR_HiPPO(N)
    Lambda, Pc, Bc, V = generate_DPLR_HiPPO(N)
    Vc = V.conj().T
    P = P.view(-1, 1)
    Pc = Pc.view(-1, 1)
    Lambda = diag(Lambda)

    A3 = V @ Lambda @ Vc - (P @ P.T)
    A4 = V @ (Lambda - Pc @ Pc.conj().T) @ Vc

    print("A2 Versus A3")
    compare_tensors(A2, A3)
    print("A2 versus A4")
    compare_tensors(A2, A4)


def test_recurrent_view_versus_convolution_view(N: int = 8, L: int = 16):
    step = 1 / L
    Lambda, P, B, _ = generate_DPLR_HiPPO(N)
    C = empty(size=(N,), dtype=complex64)
    normal_(C)

    K = kernel_DPLR(Lambda, P, P, B, C, step, L)

    A_bar, B_bar, C_bar = discretize_DPLR(Lambda, P, P, B, C, step, L)
    K2 = K_conv(A_bar, B_bar, C_bar, L=L)

    print("Comparing DPLR Kernel versus Naive Kernel")
    compare_tensors(K, K2)

    test_input = rand(size=(1, 1, L))
    y_conv = forward_pass_convolution(test_input, K)
    y_rec = forward_pass_recurrent(test_input, A_bar, B_bar, C_bar)

    print("Comparing DPLR Conv versus DPLR Recurrent")
    compare_tensors(y_conv, y_rec)


if __name__ == "__main__":
    test_recurrent_view_versus_convolution_view(L=16)
