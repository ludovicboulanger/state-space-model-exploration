from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class SequenceDataset(Dataset, ABC):
    @property
    @abstractmethod
    def num_labels(self) -> int:
        pass
