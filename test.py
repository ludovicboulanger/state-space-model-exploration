from pathlib import Path

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from config import ConfigParser, TrainingConfig
from google_speech_commands_dataset_small import SpeechCommandsDatasetSmall
from train import S4Network


def test_model(config: TrainingConfig) -> float:
    validation_dataset = SpeechCommandsDatasetSmall(
        root=config.data_root,
        download=True,
        subset="validation",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
    )
    trainer = Trainer(
        deterministic=True,
        max_epochs=config.max_epochs,
        logger=CSVLogger(
            save_dir=Path(config.save_dir) / config.run_id, name="logs", version="test"
        ),
    )

    ssm = S4Network(config, output_dim=validation_dataset.num_labels)
    ckpt_file = Path(config.save_dir) / config.run_id / "best_checkpoint.ckpt"
    trainer.test(ssm, dataloaders=validation_loader, ckpt_path=ckpt_file)
    return 0.0


if __name__ == "__main__":
    seed_everything(3221)
    run_loc = Path(
        "/Users/ludovic/Workspace/ssm-speech-processing/training-runs/fir/ssm-speech-processing/google-speech-commands-small/08270690"
    )
    config = TrainingConfig.from_json(run_loc / "training_config.json")
    config.data_root = "./data"
    config.save_dir = str(run_loc.parent)
    print(config.data_root)
    print(config.save_dir)
    test_model(config)
