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
)


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
        else:
            raise Exception(f"Invalid normalization function provided : {norm}")
