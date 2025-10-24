from typing import Optional, Tuple, Union
from torch import (
    Tensor,
    arange,
    clamp,
    complex64,
    concatenate,
    diagonal,
    einsum,
    exp,
    eye,
    diag,
    diag_embed,
    matrix_power,
    mean,
    meshgrid,
    ones_like,
    pi,
    sort,
    sum,
    sqrt,
    tril,
    view_as_complex,
    where,
    zeros,
)
from torch.fft import ifft
from torch.linalg import inv, eigh


def build_legs_matrices(N: int) -> Tuple[Tensor, Tensor]:
    # Taken from https://github.com/state-spaces/s4/blob/main/src/models/hippo/hippo.py
    Q = arange(N).double()
    cols, rows = meshgrid([Q, Q])
    M = -1 * (where(rows > cols, 2 * Q + 1, 0))
    T = sqrt(diag(2 * Q + 1))
    A = T @ M @ inv(T)
    B = diag(T).unsqueeze(dim=-1)
    return A, B


def build_low_rank_matrix(N: int, rank: int = 1) -> Tensor:
    # Taken from https://github.com/state-spaces/s4/blob/main/src/models/hippo/hippo.py
    P = sqrt(0.5 + arange(N)).unsqueeze(dim=0)
    d = P.shape[0]
    if rank > 1:
        P = concatenate((P, zeros(size=(rank - d, N))), dim=0)
    return P


def nplr_from_hippo(
    N: int, B_clip: Optional[float] = 2
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Taken from https://github.com/state-spaces/s4/blob/main/src/models/hippo/hippo.py
    A, B = build_legs_matrices(N)
    A, B = A.float(), B.float()
    P = build_low_rank_matrix(N)
    P = P.float()

    AP = A + sum(P.unsqueeze(dim=-2) * P.unsqueeze(dim=-1), dim=-3)

    # AP Should be nearly skew-symmetric
    _A = AP + AP.transpose(-1, -2)
    err = sum((_A - _A[0, 0] * eye(N)) ** 2 / N)
    if err > 1e-5:
        print("WARNING: HiPPO matrix is not skew symmetric", err)

    W_re = mean(diagonal(AP), dim=-1, keepdim=True)
    AP = AP.double()
    W_im, V = eigh(AP * 1j)
    W_im, V = W_im.to(complex64), V.to(complex64)
    W = W_re + 1j * W_im

    _, indices = sort(W.imag)
    W_sorted = W[indices]
    V_sorted = V[:, indices]

    V = V_sorted[:, : N // 2]
    W = W_sorted[: N // 2]
    assert W[-2].abs() > 1e-4, (
        "Only 1 non-zero eigenvalue allowed in diagonal part of A"
    )
    if W[-1].abs() < 1e-4:
        V[:, -1] = 0.0
        V[0, -1] = 2**-0.5
        V[1, -1] = 2**-0.5 * 1j

    _AP = diag_embed(W) @ V.conj_physical().transpose(-1, -2)
    err = sum((2 * _AP.real - AP) ** 2) / N
    if err > 1e-5:
        print("Warning: Diagonalization is not numerically precise - error : ", err)

    V_inv = V.conj().transpose(-1, -2)
    B = einsum("ij,j->i", V_inv, B.to(V))
    P = einsum("ij,...j -> ...i", V_inv, P.to(V))

    if B_clip is not None:
        B = B.real + 1j * clamp(B.imag, min=-1 * B_clip, max=B_clip)
    # W is the imaginary part of the DPLR form A = W - PP^*
    return W, P, B, V


def generate_kernel(C: Tensor) -> Tensor:
    C = view_as_complex(C)


def linear_discretization(
    A: Tensor, B: Tensor, C: Tensor, P: Tensor, Q: Tensor, step: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    D = (2.0 / step - A).reciprocal()
    R = (
        eye(n=1, dtype=A.dtype, device=A.device)
        + 2 * einsum("rhn,hn,shn->hrs", Q, D, P).real
    )
    Q_D = rea


def build_A_from_DPLR(Lambda: Tensor, P: Tensor, Q: Tensor) -> Tensor:
    return diag(Lambda) - P.view(-1, 1) @ Q.view(-1, 1).conj_physical().T


def generate_hippo_matrix(N: int) -> Tensor:
    P = sqrt(1 + 2 * arange(N))
    A = P.view(-1, 1) * P.view(1, -1)
    A = tril(A) - diag(arange(N))
    return -A


def kernel_DPLR(
    Lambda: Tensor,
    P: Tensor,
    Q: Tensor,
    B: Tensor,
    C: Tensor,
    step: Tensor,
    L: int,
) -> Tensor:
    step = step.unsqueeze(dim=0)
    omega_L = exp((-2j * pi) * (arange(L) / L)).unsqueeze(dim=-1)
    a_term = (C.conj_physical(), Q.conj_physical())
    b_term = (B, P)

    g = (2.0 / step.view(1, -1)) * ((1.0 - omega_L) / (1.0 + omega_L)).to(Lambda.device)
    c = 2.0 / (1.0 + omega_L).to(Lambda.device)

    k00 = cauchy_kernel(a_term[0] * b_term[0], g, Lambda)
    k01 = cauchy_kernel(a_term[0] * b_term[1], g, Lambda)
    k10 = cauchy_kernel(a_term[1] * b_term[0], g, Lambda)
    k11 = cauchy_kernel(a_term[1] * b_term[1], g, Lambda)
    at_roots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    output = ifft(at_roots, dim=0, n=L)
    return output


def cauchy_kernel(v: Tensor, omega: Tensor, lambd: Tensor) -> Tensor:
    omega = omega.unsqueeze(dim=1)
    v = v.transpose(dim0=-1, dim1=-2).unsqueeze(dim=0)
    lambd = lambd.transpose(dim0=-1, dim1=-2).unsqueeze(dim=0)
    return (v / (omega - lambd)).sum(dim=1)


def discretize_DPLR(
    Lambda: Tensor,
    P: Tensor,
    Q: Tensor,
    B: Tensor,
    C: Tensor,
    step: Tensor,
    L: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    B = B.unsqueeze(dim=-1)
    C = C.unsqueeze(dim=1)

    N = Lambda.shape[1]
    A = diag_embed(Lambda) - P.unsqueeze(dim=-1) @ Q.unsqueeze(
        dim=-1
    ).conj_physical().transpose(dim0=-1, dim1=-2)
    I_matrix = eye(N).unsqueeze(dim=0)

    A0 = (2.0 / step.view(-1, 1, 1)) * I_matrix + A

    D = diag_embed(1.0 / ((2.0 / step.view(-1, 1)) - Lambda))
    Qc = Q.conj_physical().unsqueeze(dim=1)
    P2 = P.unsqueeze(dim=-1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)
    A_bar = A1 @ A0
    B_bar = 2 * A1 @ B

    C_bar = C @ inv(I_matrix - matrix_power(A_bar, L)).conj_physical()
    return A_bar, B_bar, C_bar.conj_physical()


def generate_DPLR_HiPPO(N) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    A, P, B = generate_NPLR_HiPPO(N)
    S = A + P.view(-1, 1) * P.view(1, -1)
    S_diag = diag(S)
    Lambda_real = mean(S_diag) * ones_like(S_diag)
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj_physical().T @ P.type(complex64)
    B = V.conj_physical().T @ B.type(complex64)
    return Lambda_real + 1j * Lambda_imag, P, B, V


def generate_NPLR_HiPPO(N: int) -> Tuple[Tensor, Tensor, Tensor]:
    nhippo = generate_hippo_matrix(N)

    P = sqrt(arange(N) + 0.5)
    B = sqrt(2 * arange(N) + 1.0)

    return nhippo, P, B


def compare_tensors(a: Tensor, b: Tensor) -> None:
    max_error = (a - b).abs().max()
    relative_error = max_error / a.abs().max()
    print(f"Max error: {max_error}, Relative: {relative_error}")
