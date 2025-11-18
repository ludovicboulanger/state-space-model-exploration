from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class SequenceDataset(Dataset, ABC):
    def __init__(self, upsampling_factor: int, data_encoding: str) -> None:
        super().__init__()
        self._upsampling_factor = upsampling_factor
        self._data_encoding = data_encoding

    @property
    @abstractmethod
    def num_labels(self) -> int:
        pass

    @property
    def upsampling_factor(self) -> int:
        return self._upsampling_factor

    @upsampling_factor.setter
    def upsampling_factor(self, factor: int) -> None:
        self._upsampling_factor = factor

    @property
    def data_encoding(self) -> str:
        return self._data_encoding

    @data_encoding.setter
    def data_encoding(self, encoding: str) -> None:
        if encoding not in ["pcm", "pdm"]:
            raise ValueError(f"Unrecognized Data Encoding: {encoding}")
        self._data_encoding = encoding
