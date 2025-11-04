from typing import Tuple
from torch import Tensor, tensor
from torch.nn import Tanh
from torchmetrics.audio import (
    ScaleInvariantSignalNoiseRatio,
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)
from config import TrainingConfig
from modules.sequence_module import SequenceModule
from utils.loss import TemporalSpectralLoss


class RegressionModule(SequenceModule):
    def __init__(self, config: TrainingConfig, output_dim: int) -> None:
        super().__init__(config, output_dim)
        self._output_activation = Tanh()
        self._pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")
        self._stoi = ShortTimeObjectiveIntelligibility(fs=16000)
        self._si_snr = ScaleInvariantSignalNoiseRatio()
        self._loss_fn = TemporalSpectralLoss(lambda_temp=0.72, lambda_mag=0.001, lambda_phase=0.014, alpha_mag=0.5)

    def training_step(self, batch: Tuple[Tensor, ...]) -> Tensor:
        out = self._forward_pass(batch, pool=False)
        return self._compute_and_log_loss(out, batch[1], "train")

    def validation_step(self, batch: Tuple[Tensor, ...]) -> Tensor:
        out = self._forward_pass(batch, pool=False)
        loss = self._compute_and_log_loss(out, batch[1], "valid")
        self._compute_and_log_metrics(out, batch[1], "valid")
        return loss

    def test_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        input_length = batch[0].shape[1]
        for block in self._blocks:
            block.seq_len = input_length  # type: ignore

        out = self._forward_pass(batch, pool=False)
        loss = self._compute_and_log_loss(out, batch[1], "test")
        self._compute_and_log_metrics(out, batch[1], "test")

        from soundfile import write
        from pathlib import Path

        pred = out.cpu().detach().numpy().squeeze()
        target = batch[1].cpu().detach().numpy().squeeze()
        noisy = batch[0].cpu().detach().numpy().squeeze()
        save_dir = Path(self._config.save_dir) / self._config.run_id / "test-outputs"
        save_dir.mkdir(exist_ok=True)
        write(save_dir / f"{batch_idx}-prediction.wav", data=pred, samplerate=16_000)
        write(save_dir / f"{batch_idx}-target.wav", data=target, samplerate=16_000)
        write(save_dir / f"{batch_idx}-noisy.wav", data=noisy, samplerate=16_000)

        return loss

    def _compute_and_log_loss(self, predictions: Tensor, targets: Tensor, split: str) -> Tensor:
        loss = -1 * self._si_snr(predictions.transpose(-1, -2), targets.transpose(-1, -2))
        if split == "train":
            self.log(
                name=f"{split}_loss",
                value=loss.item(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
        else:
            self.log(
                name=f"{split}_loss",
                value=loss.item(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
        return loss

    def _compute_and_log_metrics(self, predictions: Tensor, targets: Tensor, split: str) -> None:
        si_snr = self._si_snr(predictions.transpose(-1, -2), targets.transpose(-1, -2))
        try:
            pesq = self._pesq(predictions.transpose(-1, -2), targets.transpose(-1, -2))
        except Exception:
            pesq = tensor(-0.5)
        try:
            stoi = self._stoi(predictions.transpose(-1, -2), targets.transpose(-1, -2))
        except Exception:
            stoi = tensor(0.0)
        self.log(name=f"{split}_si_snr", value=si_snr.item(), on_step=False, on_epoch=True, sync_dist=True)
        self.log(name=f"{split}_pesq", value=pesq.item(), on_step=False, on_epoch=True, sync_dist=True)
        self.log(name=f"{split}_stoi", value=stoi.item(), on_step=False, on_epoch=True, sync_dist=True)

    def _forward_pass(self, batch: Tuple[Tensor, ...], pool: bool) -> Tensor:
        out = super()._forward_pass(batch, pool)
        out = self._output_activation(out)
        return out
