from torch import Tensor, view_as_real, view_as_complex
from torch.cuda import is_available as is_cuda_available

try:
    from extensions.kernels.cauchy import cauchy_mult

    print("Using CUDA Kernels")
except Exception:
    print("Error Loading CUDA kernels, {e}\n Falling back to pyekops implementation.")
    from pykeops.torch import Genred

    def cauchy_mult(v: Tensor, z: Tensor, w: Tensor) -> Tensor:
        def _broadcast_dims(*tensors):
            max_dim = max([len(tensor.shape) for tensor in tensors])
            tensors = [tensor.view((1,) * (max_dim - len(tensor.shape)) + tensor.shape) for tensor in tensors]
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
