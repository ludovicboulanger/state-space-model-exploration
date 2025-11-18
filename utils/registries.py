from typing import Tuple
from torch import Tensor, arange, diag, meshgrid, pi, sqrt, stack, where, zeros
from torch.linalg import inv
from torch.nn import Module, GELU, Sigmoid, ReLU, GLU, Tanh, Identity, LayerNorm, BatchNorm1d, GroupNorm

from config import TrainingConfig
from datasets.voicebank_demand import VoiceBankDEMAND
from datasets.sequence_dataset import SequenceDataset
from datasets.speech_commands_dataset_small import SpeechCommandsDatasetSmall
from datasets.speech_commands_dataset import SpeechCommandsDataset


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
        elif norm == "group":
            return GroupNorm(num_groups=channels, num_channels=channels)
        elif norm == "none":
            return Identity()
        else:
            raise Exception(f"Invalid normalization function provided : {norm}")


class DatasetRegistry:
    @staticmethod
    def instantiate(config: TrainingConfig, split: str) -> SequenceDataset:
        if config.dataset == "gcs-sm":
            return SpeechCommandsDatasetSmall(
                root=config.data_root,
                download=False,
                subset=split,
                data_encoding=config.data_encoding,
                upsampling_factor=config.upsampling_factor,
            )

        elif config.dataset == "gcs":
            return SpeechCommandsDataset(
                root=config.data_root,
                download=False,
                subset=split,
                data_encoding=config.data_encoding,
                upsampling_factor=config.upsampling_factor,
            )
        elif config.dataset == "voicebank-28":
            return VoiceBankDEMAND(
                root=config.data_root,
                subset=split,
                speakers=28,
                data_encoding=config.data_encoding,
                upsampling_factor=config.upsampling_factor,
            )
        elif config.dataset == "voicebank-56":
            return VoiceBankDEMAND(
                root=config.data_root,
                subset=split,
                speakers=56,
                data_encoding=config.data_encoding,
                upsampling_factor=config.upsampling_factor,
            )
        else:
            raise Exception(f"Unrecognized dataset id from config: {config.dataset}")


class StateMatricesRegistry:
    @staticmethod
    def instantiate(mode: str, state_dim: int) -> Tuple[Tensor, Tensor]:
        if mode == "s4":
            """
            This functions implements the HiPPO A matrix according to equation
            (2) of S4 paper and the corresponding Theorem 2 of the HiPPO paper
            """
            q = arange(state_dim).double()
            rows, cols = meshgrid(q, q, indexing="ij")
            r = 2 * q + 1
            M = -1 * (where(rows >= cols, r, 0) - diag(q))
            T = sqrt(diag(r))
            A = T @ M @ inv(T)
            B = diag(T)
            return A, B.unsqueeze(dim=-1)
        elif mode == "lmu":
            """
            This function implements equation (2) of the LMU Paper up to the theta
            normalization.
            """
            N = arange(state_dim).view(-1, 1)
            R = 2 * N + 1

            i, j = meshgrid(N.squeeze(), N.squeeze(), indexing="ij")
            A = where(i < j, -1, (-1) ** (i - j + 1))
            A = R * A
            B = R * (-1) ** N
            return A, B
        elif mode == "fout":
            freqs = arange(state_dim // 2)
            d = stack([zeros(state_dim // 2), freqs], dim=-1).reshape(-1)[1:]
            A = pi * (-diag(d, 1) + diag(d, -1))
            B = zeros(state_dim)
            B[0::2] = 2**0.5
            B[0] = 1

            # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
            A = A - B.view(-1, 1) ** 2
            B = B.view(-1, 1)
            return A, B
        else:
            raise ValueError(f"Unrecognized state matrix mode : {mode}")
