from pathlib import Path
from typing import Dict, Tuple
from numpy import array, ndarray
from torch import Tensor, tensor

from speech_commands_dataset import SpeechCommandsDataset


class SpeechCommandsDatasetSmall(SpeechCommandsDataset):
    def __init__(
        self,
        root: str | Path,
        url: str = "speech_commands_v0.02",
        folder_in_archive: str = "SpeechCommands",
        download: bool = False,
        subset: str = "training",
    ) -> None:
        super(SpeechCommandsDatasetSmall, self).__init__(
            root=root,
            url=url,
            folder_in_archive=folder_in_archive,
            download=download,
            subset=subset,
        )
        self._indices = self._build_dataset()

    @property
    def label_to_id(self) -> Dict[str, int]:
        return {
            "down": 0,
            "go": 1,
            "left": 2,
            "no": 3,
            "off": 4,
            "on": 5,
            "right": 6,
            "stop": 7,
            "up": 8,
            "yes": 9,
        }

    @property
    def id_to_label(self) -> Dict[int, str]:
        return {v: k for k, v in self.label_to_id.items()}

    @property
    def num_labels(self) -> int:
        return len(list(self.id_to_label.keys()))

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        data, _, label, _, _ = self._dataset[self._indices[n]]
        label_as_tensor = tensor(self.label_to_id[label]).long()
        data = self._pad_data_if_needed(data)
        return data.transpose(dim0=-1, dim1=-2), label_as_tensor

    def __len__(self) -> int:
        return len(self._indices)

    def _build_dataset(self) -> ndarray:
        indicies = []
        for i in range(len(self._dataset)):
            _, _, label, _, _ = self._dataset[i]
            if label in self.label_to_id.keys():
                indicies.append(i)
        return array(indicies)
