from pathlib import Path
from typing import Tuple

from numpy.random import default_rng
from pandas import read_csv
from soundfile import read

from torch import Tensor, from_numpy, tensor

from datasets.sequence_dataset import SequenceDataset


class VoiceBankDEMAND(SequenceDataset):
    def __init__(
        self,
        root: str,
        data_encoding: str = "pcm",
        pdm_factor: int = 64,
        subset: str = "training",
        speakers: int = 28,
    ) -> None:
        super().__init__()
        if speakers == 28:
            self._root = root + "/VoiceBankDemand_28spk"
        else:
            self._root = root + "/VoiceBankDemand_56spk"
        self._encoding = data_encoding
        self._pdm_factor = pdm_factor
        self._rng = default_rng(seed=3221)

        self._subset = subset
        if subset == "training":
            self._data = read_csv(Path(self._root) / "train.csv")
            self._subfolder = "train"
        elif subset == "validation":
            self._data = read_csv(Path(self._root) / "valid.csv")
            self._subfolder = "valid"
        else:
            # All tests are done on the validation set for now. Raw measn that the set was not
            # split into 1 second segments.
            self._data = read_csv(Path(self._root) / "valid_raw.csv")
            self._subfolder = "valid_raw"

    @property
    def num_labels(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        sample_metadata = self._data.iloc[index].to_dict()
        noisy_sample_loc = f"{self._subfolder}/noisy/{sample_metadata['filename']}"
        clean_sample_loc = f"{self._subfolder}/clean/{sample_metadata['filename']}"
        noisy_data, _ = read(Path(self._root) / noisy_sample_loc)
        clean_data, _ = read(Path(self._root) / clean_sample_loc)
        return (
            from_numpy(noisy_data).view(-1, 1).float(),
            from_numpy(clean_data).view(-1, 1).float(),
            tensor(float(sample_metadata["snr_db"])).float(),
        )

    def __len__(self) -> int:
        return len(self._data)


if __name__ == "__main__":
    ds = VoiceBankDEMAND(root="./data")
    ds[19]
