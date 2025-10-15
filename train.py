from typing import Tuple
from lightning import Trainer, LightningModule
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer
from torch.nn import ModuleList, Linear
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy

from mnist_dataset import MNISTDataset
from s4_model import S4Block


class S4Network(LightningModule):
    def __init__(
        self,
        num_layers: int,
        channels: int,
        hidden_dim: int,
        seq_len: int,
        step: float,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self._inference_mode = False
        self._encoder = Linear(in_features=1, out_features=channels)
        self._blocks = ModuleList([])
        for i in range(num_layers):
            self._blocks.append(
                S4Block(
                    channels=channels,
                    hidden_dim=hidden_dim,
                    seq_len=seq_len,
                    step=step,
                    dropout_prob=dropout_prob,
                )
            )
        self._decoder = Linear(in_features=channels, out_features=10)

    @property
    def inference_mode(self) -> bool:
        return self._inference_mode

    @inference_mode.setter
    def inference_mode(self, mode: bool) -> None:
        self._inference_mode = mode
        for block in self._blocks:
            block.inference_mode = mode  # type: ignore

    def training_step(self, batch: Tuple[Tensor, ...]) -> Tensor:
        loss, accuracy = self._forward_pass(batch)
        self.log(name="train_loss", value=loss.item(), on_step=False, on_epoch=True)
        self.log(
            name="train_accuracy", value=accuracy.item(), on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch: Tuple[Tensor, ...]) -> Tensor:
        loss, accuracy = self._forward_pass(batch)
        self.log(name="valid_loss", value=loss.item(), on_step=False, on_epoch=True)
        self.log(
            name="valid_accuracy", value=accuracy.item(), on_step=False, on_epoch=True
        )
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = Adam(params=self._blocks.parameters(), lr=1e-3)
        return optimizer

    def _forward_pass(self, batch: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
        x, y = batch

        y_hat = self._encoder(x)
        for block in self._blocks:
            y_hat = block(y_hat)
        y_hat = self._decoder(y_hat).mean(dim=1)

        loss = cross_entropy(y_hat, y)
        acc = Accuracy(task="multiclass", num_classes=10)(y_hat.argmax(dim=-1), y)
        return loss, acc


def train_model() -> float:
    ssm = S4Network(
        num_layers=4,
        channels=16,
        hidden_dim=256,
        seq_len=int(28**2),
        step=1 / 28**2,
        dropout_prob=0.0,
    )

    training_dataset = MNISTDataset(
        root="./data", download=True, subset="training", sequential=True
    )
    validation_dataset = MNISTDataset(
        root="./data", download=True, subset="validation", sequential=True
    )
    training_loader = DataLoader(training_dataset, batch_size=8, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)
    trainer = Trainer(accelerator="cpu")
    trainer.fit(
        ssm,
        train_dataloaders=training_loader,
        val_dataloaders=validation_loader,
    )
    return 0.0


if __name__ == "__main__":
    train_model()
