from torch import allclose, rand, view_as_complex
from s4 import S4Kernel, S4
from official.models.s4.s4 import FFTConv, SSMKernelDPLR


def test_bench_kernel() -> None:
    channels = 8
    state_dim = 16
    n_ssms = 2
    min_dt = 1e-4
    max_dt = 1e-1
    seq_len = 16000

    s4 = S4Kernel(
        channels=channels,
        n_ssms=n_ssms,
        state_dim=state_dim,
        seq_len=seq_len,
        min_dt=min_dt,
        max_dt=max_dt,
    )

    official_kwargs = {
        "channels": 1,
        "d_model": channels,
        "d_state": state_dim,
        "deterministic": False,
        "dt_max": max_dt,
        "dt_min": min_dt,
        "dt_transform": "softplus",
        "init": "legs",
        "l_max": seq_len,
        "lr": {"dt": 0.001, "A": 0.001, "B": 0.001},
        "measure": None,
        "n_ssm": n_ssms,
        "rank": 1,
        "verbose": True,
        "wd": 0.0,
    }
    s4_official = SSMKernelDPLR(**official_kwargs)
    print("Both models initialized!")

    # Make sure the two models start with the same parameters
    s4_official.inv_dt.data = s4._dt.data
    s4_official.C.data = s4._C.data

    print("-----------------------------------------------------------------")
    print("TEST : Parameters are the same at initialization")
    print("-----------------------------------------------------------------")
    A_real_is_good = allclose(s4._A_real, s4_official.A_real)  # type: ignore
    A_imag_is_good = allclose(s4._A_imag, s4_official.A_imag)  # type: ignore
    P_is_good = allclose(s4._P, s4_official.P)  # type: ignore
    B_is_good = allclose(s4._B, s4_official.B)  # type: ignore
    print("\t A_real : ", A_real_is_good)
    print("\t A_imag : ", A_imag_is_good)
    print("\t P : ", P_is_good)
    print("\t B : ", B_is_good)
    print("-----------------------------------------------------------------")
    print("\n")
    print("-----------------------------------------------------------------")
    print("TEST : step_params should be the same")
    print("-----------------------------------------------------------------")
    s4_official._setup_state()
    official_step_params = s4_official.step_params
    step_params = s4._compute_linear_step_terms()
    for k in official_step_params.keys():
        param_is_good = allclose(official_step_params[k], step_params[k])
        print(f"\t {k} : ", param_is_good)
    print("-----------------------------------------------------------------")
    print("\n")
    print("-----------------------------------------------------------------")
    print("TEST : A_bar and B_bar should be the same")
    print("-----------------------------------------------------------------")
    A_bar_official, B_bar_official = s4_official._setup_state()
    A_bar, B_bar = s4._discretize_A_and_B()
    A_bar_is_good = allclose(A_bar, A_bar_official)  # type: ignore
    B_bar_is_good = allclose(B_bar, B_bar_official)  # type: ignore
    print("\t A_bar : ", A_bar_is_good)
    print("\t B_bar : ", B_bar_is_good)
    print("-----------------------------------------------------------------")
    print("\n")
    print("-----------------------------------------------------------------")
    print("TEST : C_tilde should be the same")
    print("-----------------------------------------------------------------")
    s4_official._setup_C(L=seq_len)
    C_tilde_official = view_as_complex(s4_official.C.data)
    C_tilde = s4._compute_C_tilde()
    C_tilde_is_good = allclose(C_tilde, C_tilde_official)  # type: ignore
    print("\t C_tilde : ", C_tilde_is_good)
    print("-----------------------------------------------------------------")
    print("\n")
    print("-----------------------------------------------------------------")
    print("TEST : Kernels should be the same")
    print("-----------------------------------------------------------------")
    k_official, _ = s4_official(L=seq_len)
    k = s4()
    k_is_good = allclose(k, k_official, atol=1e-6)  # type: ignore
    print("\t Kernel : ", k_is_good)
    print("-----------------------------------------------------------------")

    print("Test Suite Done!")


def test_bench_layer() -> None:
    layer_args = {
        "swap_channels": False,
        "bidirectional": False,  # Unsupported for now.
        "activation": None,
        "transposed": True,
        "dropout": 0.0,
        "tie_dropout": False,
        "drop_kernel": 0.0,
        "mode": "dplr",
        "kernel": None,
    }
    kernel_args = {
        "channels": 1,
        "d_model": 16,
        "d_state": 32,
        "deterministic": False,
        "dt_max": 0.1,
        "dt_min": 0.0001,
        "dt_transform": "softplus",
        "init": "legs",
        "l_max": 16000,
        "lr": {"dt": 0.001, "A": 0.001, "B": 0.001},
        "measure": None,
        "n_ssm": 2,
        "rank": 1,
        "verbose": True,
        "wd": 0.0,
    }
    s4 = S4(
        channels=kernel_args["d_model"],
        n_ssms=kernel_args["n_ssm"],
        state_dim=kernel_args["d_state"],
        seq_len=kernel_args["l_max"],
        min_dt=kernel_args["dt_min"],
        max_dt=kernel_args["dt_max"],
        p_kernel_dropout=layer_args["drop_kernel"],
    )
    s4_official = FFTConv(**layer_args, **kernel_args)

    # Make sure the two models start with the same parameters
    s4_official.kernel.inv_dt.data = s4._kernel._dt.data
    s4_official.kernel.C.data = s4._kernel._C.data
    s4_official.D.data = s4._D.data.unsqueeze(dim=0)

    print("Both models initialized!")
    print("\n")
    print("-----------------------------------------------------------------")
    print("TEST : S4 and FFTConv should output the same for a given input")
    print("-----------------------------------------------------------------")
    u = rand(size=(2, kernel_args["d_model"], kernel_args["l_max"])).float()
    output_fft_conv, _ = s4_official(x=u)
    output_s4 = s4(u=u)
    print(output_s4.abs().mean(), output_fft_conv.abs().mean())
    check_close(output_fft_conv, output_s4)
    print("-----------------------------------------------------------------")


def check_close(a, b):
    abs_err = (a - b).abs()
    rel_err = abs_err / (b.abs() + 1e-12)
    print(f"\t max abs: {abs_err.max():.3e}, mean abs: {abs_err.mean():.3e}")
    print(f"\t max rel: {rel_err.max():.3e}, mean rel: {rel_err.mean():.3e}")


if __name__ == "__main__":
    from numpy.random import seed as set_np_seed
    from random import seed as set_random_seed
    from torch import manual_seed as set_torch_seed

    set_torch_seed(3221)
    set_random_seed(3221)
    set_np_seed(3221)

    test_bench_kernel()
    test_bench_layer()
