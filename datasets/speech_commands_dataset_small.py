from pathlib import Path
from typing import Dict, Tuple
from numpy import array, ndarray
from torch import Tensor
from tqdm import tqdm

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
        upsampling_factor: int = 1,
    ) -> None:
        super(SpeechCommandsDatasetSmall, self).__init__(
            root=root,
            url=url,
            folder_in_archive=folder_in_archive,
            download=download,
            subset=subset,
            data_encoding=data_encoding,
            upsampling_factor=upsampling_factor,
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

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        x, y = super().__getitem__(self._indices[n])
        return x, y

    def __len__(self) -> int:
        return len(self._indices)

    def _build_dataset(self) -> ndarray:
        indicies = []
        for i in tqdm(range(len(self._dataset)), desc=f"Creating {self._subset.title()} Dataset"):
            _, _, label, _, _ = self._dataset[i]
            if label in self.label_to_id.keys():
                indicies.append(i)
        return array(indicies)
