from pathlib import Path
from typing import Dict, Tuple
from numpy import array, ndarray
from torch import Tensor

from datasets.speech_commands_dataset import SpeechCommandsDataset


class SpeechCommandsDatasetSmall(SpeechCommandsDataset):
    def __init__(
        self,
        root: str | Path,
        url: str = "speech_commands_v0.02",
        folder_in_archive: str = "SpeechCommands",
        download: bool = False,
        subset: str = "training",
        data_encoding: str = "pcm",
        pdm_factor: int = 64,
    ) -> None:
        super(SpeechCommandsDatasetSmall, self).__init__(
            root=root,
            url=url,
            folder_in_archive=folder_in_archive,
            download=download,
            subset=subset,
        )
        self._indices = self._build_dataset()
        self._data_encoding = data_encoding
        self._pdm_factor = pdm_factor

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

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        return super()[self._indices[n]]

    def __len__(self) -> int:
        return len(self._indices)

    def _build_dataset(self) -> ndarray:
        indicies = []
        for i in range(len(self._dataset)):
            _, _, label, _, _ = self._dataset[i]
            if label in self.label_to_id.keys():
                indicies.append(i)
        return array(indicies)
