from typing import Dict, Optional, Tuple
from einops import rearrange

from numpy import log as nplog
from torch import (
    all,
    Tensor,
    arange,
    clamp,
    clamp_min,
    complex64,
    concatenate,
    diag,
    diag_embed,
    diagonal,
    dtype as TorchType,
    einsum,
    exp,
    eye,
    float32,
    log,
    mean,
    meshgrid,
    no_grad,
    pi,
    rand,
    randn,
    sort,
    sum,
    sqrt,
    view_as_complex,
    view_as_real,
    where,
    zeros,
)
from torch.fft import rfft, irfft
from torch.nn import Dropout, Identity, GLU, Sequential, Linear, BatchNorm1d, GroupNorm
from torch.nn.functional import softplus
from torch.linalg import inv, eigh, solve, matrix_power
from torch.nn import Module, Parameter

from utils.registries import ActivationRegistry, NormRegistry
from utils.computations import cauchy_mult


class S4Block(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_ssms: int,
        state_dim: int,
        seq_len: int,
        min_dt: float,
        max_dt: float,
        tie_dt: bool = True,
        clip_B: Optional[float] = 2.0,
        residual: bool = True,
        dtype: TorchType = float32,
        p_kernel_dropout: float = 0.0,
        p_block_dropout: float = 0.0,
        norm: str = "batch",
        prenorm: bool = True,
        layer_activation: str = "none",
        final_activation: str = "none",
    ) -> None:
        super().__init__()
        self._s4_layer = S4(
            in_channels,
            n_ssms,
            state_dim,
            seq_len,
            min_dt,
            max_dt,
            tie_dt,
            clip_B,
            residual,
            dtype,
            p_kernel_dropout,
        )
        if p_block_dropout > 0.0:
            self._block_dropout = Dropout(p_block_dropout)
        else:
            self._block_dropout = Identity()

        self._prenorm = prenorm
        self._norm = NormRegistry.instantiate(norm=norm, channels=in_channels)

        self._layer_activation = ActivationRegistry.instantiate(activation=layer_activation)
        output_norm = ActivationRegistry.instantiate(activation=final_activation)
        if isinstance(output_norm, GLU):
            _out_features = 2 * out_channels
        else:
            _out_features = out_channels
        self._output_module = Sequential(
            Linear(in_features=in_channels, out_features=_out_features),
            output_norm,
        )
        self._residual = residual
        if residual and out_channels != in_channels:
            self._residual_connection = Linear(in_features=in_channels, out_features=out_channels)
        elif residual and out_channels == in_channels:
            self._residual_connection = Identity()

    @property
    def seq_len(self) -> int:
        return self._s4_layer.seq_len

    @seq_len.setter
    def seq_len(self, seq_len: int) -> None:
        self._s4_layer.seq_len = seq_len

    def forward(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        if self._prenorm:
            u = self._apply_normalization(u)

        y = self._s4_layer(u)
        layer_output = y
        y = self._layer_activation(y)
        y = self._block_dropout(y)
        y = self._output_module(y)

        if self._residual:
            y = y + self._residual_connection(u)

        if not self._prenorm:
            y = self._apply_normalization(y)

        return y, layer_output

    def _apply_normalization(self, u: Tensor) -> Tensor:
        if isinstance(self._norm, BatchNorm1d) or isinstance(self._norm, GroupNorm):
            y = u.transpose(dim0=-1, dim1=-2)
            y = self._norm(y)
            y = y.transpose(dim0=-1, dim1=-2)
        else:
            y = self._norm(u)
        return y


class S4(Module):
    def __init__(
        self,
        channels: int,
        n_ssms: int,
        state_dim: int,
        seq_len: int,
        min_dt: float,
        max_dt: float,
        tie_dt: bool = True,
        clip_B: Optional[float] = 2.0,
        use_D: bool = True,
        dtype: TorchType = float32,
        p_kernel_dropout: float = 0.0,
    ) -> None:
        super(S4, self).__init__()
        self._kernel = S4Kernel(
            channels,
            n_ssms,
            state_dim,
            seq_len,
            min_dt,
            max_dt,
            tie_dt,
            clip_B,
            dtype,
        )
        self._seq_len = seq_len
        if use_D:
            self._D = Parameter(data=randn(size=(channels,)), requires_grad=True)
        else:
            self._D = Parameter(data=zeros(size=(channels,)), requires_grad=False)
        self._kernel_dropout = Dropout(p=p_kernel_dropout) if p_kernel_dropout > 0.0 else Identity()

    @property
    def seq_len(self) -> int:
        return self._kernel.seq_len

    @seq_len.setter
    def seq_len(self, seq_len: int) -> None:
        self._seq_len = seq_len
        self._kernel.seq_len = seq_len

    def forward(self, u: Tensor) -> Tensor:
        y = self._kernel_forward_pass(u).reshape(u.shape)
        y = y + einsum("blh,h->blh", u, self._D)
        return y

    def _kernel_forward_pass(self, u: Tensor) -> Tensor:
        u = u.transpose(-1, -2)
        kernel = self._kernel()
        kernel = self._kernel_dropout(kernel)

        k_fft = rfft(kernel, n=2 * self._seq_len)
        u_fft = rfft(u, n=2 * self._seq_len)
        y_fft = einsum("bhl,chl->bchl", u_fft, k_fft)

        y = irfft(y_fft, n=2 * self._seq_len)[..., : self._seq_len]
        return y.transpose(-1, -2)


class S4Kernel(Module):
    def __init__(
        self,
        channels: int,
        n_ssms: int,
        state_dim: int,
        seq_len: int,
        min_dt: float,
        max_dt: float,
        tie_dt: bool = True,
        clip_B: Optional[float] = 2.0,
        dtype: TorchType = float32,
    ) -> None:
        super(S4Kernel, self).__init__()
        self._channels = channels
        if n_ssms == -1:
            self._n_ssms = channels
        else:
            self._n_ssms = n_ssms
        assert self._channels % self._n_ssms == 0, (
            f"Error: n_ssms does not divide channels. n_ssms = {self._n_ssms} ; channels = {self._channels}"
        )
        self._state_dim = state_dim
        self._seq_len = seq_len
        self._min_dt = min_dt
        self._max_dt = max_dt
        self._tie_dt = tie_dt
        self._clip_B = clip_B
        self._dtype = dtype

        dt = Parameter(data=self._init_dt(), requires_grad=True)
        A, P, B, _ = self._init_dplr_parameters()
        C = self._init_C_matrix()
        # TODO: Look at official register_params in SSMKernelDiag for reference
        self._register_parameters(A, P, B, C, dt)

        # Assume Conjugate symmetry in the parameters
        self._state_dim = self._state_dim // 2
        self._scale_factor = 1.0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @seq_len.setter
    def seq_len(self, seq_len: int) -> None:
        self._seq_len = seq_len

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, scale_factor: float) -> None:
        self._scale_factor = scale_factor

    def forward(self) -> Tensor:
        """
        Compute the S4 Kernel using the Bilinear Transform to discretize parameters. This function
        implements the Appendix C.2 and C.3
        """
        C_t = self._compute_C_tilde()
        A, B, _, P, Q, dt = self._get_parameters()
        omega = exp(-2j * pi * arange(self._seq_len // 2 + 1) / self._seq_len).to(A)
        z = 2 * (1 - omega) / (1 + omega)

        A = A * dt
        B = concatenate((B, P), dim=-3)
        C = concatenate((C_t, Q), dim=-3)
        v = B.unsqueeze(dim=-3) * C.unsqueeze(dim=-4)
        v = v * dt

        r = cauchy_mult(v, z, A)
        K = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        K = K * 2 / (1 + omega)
        kernel = irfft(K, n=self._seq_len)

        return kernel[-1, :, :, :]

    @no_grad()
    def _compute_C_tilde(self) -> Tensor:
        C = view_as_complex(self._C)
        A_bar, _ = self._discretize_A_and_B()
        # Note: Official Implementation uses their own function
        dA_L = matrix_power(A_bar, self._seq_len)

        C_t = concatenate((C, C.conj_physical()), dim=-1)
        prod = einsum("hmn,chn -> chm", dA_L.transpose(-1, -2), C_t)
        C_t = C_t - prod
        C_t = C_t[..., : self._state_dim]
        return C_t

    def _discretize_A_and_B(self) -> Tuple[Tensor, Tensor]:
        C = view_as_complex(self._C)
        step_parameters = self._compute_linear_step_terms()
        state = eye(2 * self._state_dim, dtype=C.dtype, device=C.device).unsqueeze(dim=-2)
        A_bar = self._compute_new_state(step_parameters, x=state)
        A_bar = rearrange(A_bar, "n h m -> h m n")
        u = C.new_ones(size=(self._channels,))
        B_bar = self._compute_new_state(step_parameters, u=u)
        B_bar = concatenate((B_bar, B_bar.conj_physical()), dim=-1)
        B_bar = rearrange(B_bar, "1 h n -> h n")
        return A_bar, B_bar

    def _compute_new_state(
        self,
        step_parameters: Dict[str, Tensor],
        x: Optional[Tensor] = None,
        u: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Function to compute the new state given the previous state x and the input u at time t. When
        The input u is 0.0 and the previous state x is I, then the function returns the A_bar matrix.
        On the other hand, when the previous state x is None and the input u is I, then the function
        returns the B_bar matrix.
        """
        C = view_as_complex(self._C)

        if u is None:
            u = zeros(size=(self._channels,), dtype=C.dtype, device=C.device)
        if x is None:
            x = zeros(size=(self._channels, self._state_dim), dtype=C.dtype, device=C.device)

        if x.shape[-1] == self._state_dim:

            def contract_fn(p: Tensor, x: Tensor, y: Tensor) -> Tensor:
                p = concatenate((p, p.conj_physical()), dim=-1)
                x = concatenate((x, x.conj_physical()), dim=-1)
                y = concatenate((y, y.conj_physical()), dim=-1)
                return einsum("rhn,rhm,...hm -> ...hn", p, x, y)[..., : self._state_dim]
        else:
            assert x.shape[-1] == 2 * self._state_dim
            step_parameters = {p: concatenate((v, v.conj_physical()), dim=-1) for p, v in step_parameters.items()}

            def contract_fn(p: Tensor, x: Tensor, y: Tensor) -> Tensor:
                return einsum("rhn,rhm,...hm -> ...hn", p, x, y)

        D = step_parameters["D"]
        E = step_parameters["E"]
        R = step_parameters["R"]
        P = step_parameters["P"]
        Q = step_parameters["Q"]
        B = step_parameters["B"]

        # Computes the A_bar * x_{k-1} + B_bar * u_k
        x = E * x - contract_fn(P, Q, x)
        x = x + 2.0 * B * u.unsqueeze(dim=-1)
        x = D * (x - contract_fn(P, R, x))
        return x

    def _compute_linear_step_terms(self) -> Dict[str, Tensor]:
        """
        This function precomputes certain terms from the discretization. Particularly, it computes the
        D term and R = (I + Q^*D@P)^-1@Q^*@D and E = (2 / dt)*I + Λ
        """
        A, B, _, P, Q, dt = self._get_parameters()
        # Compute the D term in Appendix C.2
        D = (2.0 / dt - A).reciprocal()

        # Compute the (I + Q^*DP)^-1Q^*D term in Appendix C.2.
        # The .real and *2 result from the fact that conjugate symmetry is assumed.
        # So only take the real part, but *2 the result to compensate
        Id = eye(1, dtype=A.dtype, device=A.device)
        R = Id + 2 * einsum("rhn,hn,shn -> hrs", Q, D, P).real
        Q_D = (Q * D).transpose(dim0=-3, dim1=-2)
        R = solve(R, Q_D).transpose(dim0=-3, dim1=-2)

        # Compute the E term
        E = (2 / dt) + A
        return {"D": D, "R": R, "P": P, "Q": Q, "B": B, "E": E}

    def _init_C_matrix(self) -> Tensor:
        return randn(size=(1, self._channels, self._state_dim // 2), dtype=complex64)

    def _init_dplr_parameters(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        A, B = self._build_hippo_legs()
        P = self._build_lowrank_correction_term()
        A, B, P = A.float(), B.float(), P.float()

        # Add the lowrank. A = VΛV^-1 - PQ^*
        #                  V + PQ^* = VΛV^-1
        # Appendix C.1 LegS
        AP = A + sum(P.unsqueeze(dim=-2) * P.unsqueeze(dim=-1), dim=-3)

        # Sanity check for skew-symmetry
        _A = AP + AP.transpose(-1, -2)
        err = sum((_A - _A[0, 0] * eye(self._state_dim) ** 2) / self._state_dim)
        if err > 1e-5:
            print("WARNING: HiPPO matrix is not Skew-Symmetric. Error = ", err)

        # By adding correction term, we get AP = cI + S. S is skew-symmetric. AP is diagonalizable
        # by the same matrix as S. We can take advantage of this to compute the eigenvalues.
        W_re = mean(diagonal(AP), dim=-1, keepdim=True)
        AP = AP.double()
        W_im, V = eigh(AP * -1j)
        W_im, V = W_im.to(complex64), V.to(complex64)

        # This is the Λ term in the NPLR equation.
        W = W_re + 1j * W_im

        # Account for conjugate symmetry and keep only half of the conjugate pairs
        _, indices = sort(W.imag)
        W = W[indices][: self._state_dim // 2]
        V = V[:, indices][:, : self._state_dim // 2]

        _AP = V @ diag_embed(W) @ V.conj_physical().transpose(dim0=-1, dim1=-2)
        err = sum(((2 * _AP.real - AP) ** 2) / self._state_dim)
        if err > 1e-5:
            print("WARNING: Λ is not numerically precise. Error = ", err)

        V_inv = V.conj().transpose(-1, -2)
        B = einsum("ij, j -> i", V_inv, B.to(V))
        P = einsum("ij, ...j -> ...i", V_inv, P.to(V))

        if self._clip_B is not None:
            B = B.real + 1j * clamp(B.imag, min=-self._clip_B, max=self._clip_B)

        # W is in fact Λ and will be called A in the future for ease of reading.
        # We broadcast the parameters to have n_ssms unique copies.
        W = W.unsqueeze(dim=0).repeat(self._n_ssms, 1)  # (S, N)
        P = P.unsqueeze(dim=1).repeat(1, self._n_ssms, 1)  # (1, S, N)
        B = B.reshape(shape=(1, 1, -1)).repeat(1, self._n_ssms, 1)  # (1, S, N)
        V = V.unsqueeze(dim=0).repeat(self._n_ssms, 1, 1)  # (S, N, M)
        return W, P, B, V

    def _init_dt(self) -> Tensor:
        if self._tie_dt:
            shape = (self._channels, 1)
        else:
            # Assume conjugate symmetry in state so only consider the first half
            shape = (self._channels, self._state_dim // 2)
        # Log initialize dt in the same way as in How to Train Your HiPPO
        inv_dt = rand(size=shape, dtype=self._dtype)
        inv_dt = (nplog(self._max_dt) - nplog(self._min_dt)) * inv_dt
        inv_dt += nplog(self._min_dt)
        # Inverse Softplus initialize like in the official codebase
        inv_dt = log(exp(clamp_min(exp(inv_dt), min=1e-4)) - 1)
        return inv_dt

    def _register_parameters(self, A: Tensor, P: Tensor, B: Tensor, C: Tensor, dt: Tensor) -> None:
        # Check that diagonal part has negative real and imag part
        # (allow some tolerance for numerical precision on real part
        # since it may be constructed by a diagonalization)
        assert all(A.real < 1e-4) and all(A.imag <= 0.0)

        # The transformations on the A parameter and dt parameter come from the official repo
        self._A_real = Parameter(data=log(clamp_min(-1 * A.real, min=1e-4)), requires_grad=True)
        self._A_imag = Parameter(data=clamp_min(-1 * A.imag, min=1e-4), requires_grad=True)
        self._P = Parameter(data=view_as_real(P), requires_grad=True)
        self._B = Parameter(data=view_as_real(B), requires_grad=True)
        self._C = Parameter(data=view_as_real(C.conj_physical()), requires_grad=True)
        self._dt = Parameter(data=dt, requires_grad=True)

    def _get_parameters(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        A = -1 * exp(self._A_real) - 1j * self._A_imag
        B = view_as_complex(self._B)
        C = view_as_complex(self._C)
        dt = softplus(self._dt) / self._scale_factor
        P = view_as_complex(self._P)
        Q = P.conj_physical()

        A = A.repeat(self._channels // self._n_ssms, 1)
        B = B.repeat(1, self._channels // self._n_ssms, 1)
        P = P.repeat(1, self._channels // self._n_ssms, 1)
        Q = Q.repeat(1, self._channels // self._n_ssms, 1)

        return A, B, C, P, Q, dt

    def _build_hippo_legs(self) -> Tuple[Tensor, Tensor]:
        """
        This functions implements the HiPPO A matrix according to equation (2) of S4
        paper and the corresponding Theorem 2 of the HiPPO paper
        """
        q = arange(self._state_dim).double()
        rows, cols = meshgrid(q, q)
        r = 2 * q + 1
        M = -1 * (where(rows >= cols, r, 0) - diag(q))
        T = sqrt(diag(r))
        A = T @ M @ inv(T)
        B = diag(T)
        return A, B

    def _build_lowrank_correction_term(self) -> Tensor:
        """
        Implements the NPLR correction term according to Appendix C.1 in the S4 paper.
        More precisely, this implements the LegS correction term.
        """
        return sqrt(0.5 + arange(self._state_dim)).unsqueeze(dim=0)
