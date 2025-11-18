from typing import Tuple
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
from config import TrainingConfig
from modules.sequence_module import SequenceModule


class ClassificationModule(SequenceModule):
    def __init__(self, config: TrainingConfig, output_dim: int) -> None:
        super().__init__(config, output_dim)
        self._accuracy = Accuracy(task="multiclass", num_classes=output_dim)
        self._cross_entropy = CrossEntropyLoss()

    def training_step(self, batch: Tuple[Tensor, ...]) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, ...]) -> Tensor:
        return self._step(batch, "valid")

    def test_step(self, batch: Tuple[Tensor, ...]) -> Tensor:
        return self._step(batch, "test")

    def _step(self, batch: Tuple[Tensor, ...], subset: str) -> Tensor:
        out = self._forward_pass(batch, pool=True)
        loss = self._cross_entropy(out, batch[1])
        accuracy = self._accuracy(out, batch[1])
        self.log(
            name=f"{subset}_loss",
            value=loss.item(),
            on_step=subset == "test",
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            name=f"{subset}_accuracy",
            value=accuracy.item(),
            on_step=subset == "test",
            on_epoch=True,
            sync_dist=True,
        )
        return loss
