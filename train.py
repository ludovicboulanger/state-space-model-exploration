from pathlib import Path
from typing import Any, Dict, Tuple
from lightning import Trainer, LightningModule
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import ModuleList, Linear
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy

from config import ConfigParser, TrainingConfig
from mnist_dataset import MNISTDataset
from s4_model import S4Block


class S4Network(LightningModule):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self._inference_mode = False
        self._encoder = Linear(in_features=1, out_features=config.channel_dim)
        self._blocks = ModuleList([])
        for i in range(config.num_layers):
            self._blocks.append(
                S4Block(
                    channels=config.channel_dim,
                    hidden_dim=config.hidden_dim,
                    seq_len=config.seq_len,
                    step=config.step,
                    dropout_prob=config.dropout_prob,
                )
            )
        self._decoder = Linear(in_features=config.channel_dim, out_features=10)
        self._accuracy = Accuracy(task="multiclass", num_classes=10)
        self._config = config

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

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore
        optimizer = Adam(params=self._blocks.parameters(), lr=config.lr)
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=config.lr_decay,
            patience=config.lr_decay_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "valid_loss",
                "strict": True,
            },
        }

    def _forward_pass(self, batch: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
        x, y = batch

        y_hat = self._encoder(x)
        for block in self._blocks:
            y_hat = block(y_hat)
        y_hat = self._decoder(y_hat)
        logits = y_hat[:, -1, :]

        loss = cross_entropy(logits, y)
        acc = self._accuracy(logits.argmax(dim=-1), y)
        return loss, acc


def train_model(config: TrainingConfig) -> float:
    (Path(config.save_dir) / config.run_id).mkdir(parents=True, exist_ok=True)
    ssm = S4Network(config)
    summary = ModelSummary(ssm, max_depth=1000)

    with open(
        Path(config.save_dir) / config.run_id / "model_summary.txt", "w"
    ) as ostream:
        ostream.write(str(summary))

    training_dataset = MNISTDataset(
        root="./data", download=True, subset="training", sequential=True
    )
    validation_dataset = MNISTDataset(
        root="./data", download=True, subset="validation", sequential=True
    )
    training_loader = DataLoader(training_dataset, batch_size=8, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)
    trainer = Trainer(
        deterministic=True,
        max_epochs=config.max_epochs,
        logger=CSVLogger(save_dir=Path(config.save_dir) / config.run_id, name="logs"),
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(
                monitor="valid_loss",
                patience=config.early_stop_patience,
            ),
            ModelCheckpoint(
                dirpath=Path(config.save_dir) / config.run_id,
                filename="last_checkpoint",
                every_n_epochs=1,
                save_on_train_epoch_end=True,
                enable_version_counter=False,
            ),
            ModelCheckpoint(
                dirpath=Path(config.save_dir) / config.run_id,
                filename="best_checkpoint",
                monitor="valid_loss",
                mode="min",
                save_last=False,
                enable_version_counter=False,
            ),
        ],
    )
    trainer.fit(
        ssm,
        train_dataloaders=training_loader,
        val_dataloaders=validation_loader,
    )
    return 0.0


if __name__ == "__main__":
    config = ConfigParser.from_cli_args()
    train_model(config=config)
