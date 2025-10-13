from pathlib import Path
from typing import Tuple
from torch import Tensor, tensor
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        url: str = "speech_commands_v0.02",
        folder_in_archive: str = "SpeechCommands",
        download: bool = False,
        subset: str | None = None,
    ) -> None:
        super(SpeechCommandsDataset, self).__init__()
        self._dataset = SPEECHCOMMANDS(root, url, folder_in_archive, download, subset)
        self._subset = subset
        self._label_to_id = {
            "backward": 0,
            "bed": 1,
            "bird": 2,
            "cat": 3,
            "dog": 4,
            "down": 5,
            "eight": 6,
            "five": 7,
            "follow": 8,
            "forward": 9,
            "four": 10,
            "go": 11,
            "happy": 12,
            "house": 13,
            "learn": 14,
            "left": 15,
            "marvin": 16,
            "nine": 17,
            "no": 18,
            "off": 19,
            "on": 20,
            "one": 21,
            "right": 22,
            "seven": 23,
            "sheila": 24,
            "six": 25,
            "stop": 26,
            "three": 27,
            "tree": 28,
            "two": 29,
            "up": 30,
            "visual": 31,
            "wow": 32,
            "yes": 33,
            "zero": 34,
        }
        self._id_to_label = {v: k for k, v in self._label_to_id.items()}

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        data, _, label, _, _ = self._dataset[n]
        label_as_tensor = tensor(self._label_to_id[label]).long()
        return data, label_as_tensor

    def __len__(self) -> int:
        return len(self._dataset)
