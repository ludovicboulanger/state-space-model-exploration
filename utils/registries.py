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
                download=True,
                subset=split,
                data_encoding=config.data_encoding,
                pdm_factor=config.pdm_factor,
            )

        elif config.dataset == "gcs":
            return SpeechCommandsDataset(
                root=config.data_root,
                download=True,
                subset=split,
                data_encoding=config.data_encoding,
                pdm_factor=config.pdm_factor,
            )
        elif config.dataset == "voicebank-28":
            return VoiceBankDEMAND(
                root=config.data_root,
                subset=split,
                speakers=28,
                data_encoding=config.data_encoding,
                pdm_factor=config.pdm_factor,
            )
        elif config.dataset == "voicebank-56":
            return VoiceBankDEMAND(
                root=config.data_root,
                subset=split,
                speakers=56,
                data_encoding=config.data_encoding,
                pdm_factor=config.pdm_factor,
            )
        else:
            raise Exception(f"Unrecognized dataset id from config: {config.dataset}")
