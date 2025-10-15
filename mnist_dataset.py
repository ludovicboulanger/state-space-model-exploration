from pathlib import Path
from typing import Dict, Tuple
from torch import Tensor, arange, randperm, tensor
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class MNISTDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        subset: str = "training",
        sequential: bool = False,
    ) -> None:
        super(MNISTDataset, self).__init__()
        self._subset = subset
        self._sequential = sequential
        if subset in ["training", "validation"]:
            self._dataset = MNIST(
                root, train=True, download=download, transform=ToTensor()
            )
            self._indices = self._split_training_set()
        else:
            self._dataset = MNIST(root, train=False, download=download)
            self._indices = arange(len(self._dataset))

    @property
    def label_to_id(self) -> Dict[str, int]:
        return {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
        }

    @property
    def id_to_label(self) -> Dict[int, str]:
        return {v: k for k, v in self.label_to_id.items()}

    @property
    def num_labels(self) -> int:
        return len(list(self.id_to_label.keys()))

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        data, label = self._dataset[int(self._indices[n].item())]
        if self._sequential:
            data = data.view(-1, 1)
        return data, tensor(label).long()

    def __len__(self) -> int:
        return int(self._indices.shape[-1])

    def _split_training_set(self) -> Tensor:
        indices = randperm(len(self._dataset))
        if self._subset == "training":
            return indices[: int(0.8 * indices.shape[-1])]
        else:
            return indices[int(0.8 * indices.shape[-1]) :]
