from typing import Tuple
from torch import atan2, sin, cos, Tensor, hann_window, stft, zeros_like
from torch.nn import Module
from torch.nn.functional import mse_loss
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio


class TemporalSpectralLoss(Module):
    def __init__(
        self,
        lambda_temp: float = 1.0,
        lambda_mag: float = 0.0,
        lambda_phase: float = 0.0,
        alpha_mag: float = 1.0,
    ) -> None:
        super().__init__()
        self._lambda_temp = lambda_temp
        self._lambda_mag = lambda_mag
        self._alpha_mag = lambda_mag
        self._lambda_phase = lambda_phase
        assert lambda_temp >= 0 and lambda_temp <= 1.0, "Lambda Temporal must be between 0 and 1"
        assert lambda_mag >= 0 and lambda_mag <= 1.0, "Lambda Magnitude must be between 0 and 1"
        assert alpha_mag >= 0 and alpha_mag <= 1.0, "Alpha Magnitude must be between 0 and 1"
        assert lambda_phase >= 0 and lambda_phase <= 1.0, "Lambda Phase must be between 0 and 1"

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # This loss function uses un-deterministic operations. Therefore, we need to send the inputs to CPU
        device = predictions.device
        predictions = predictions.to("cpu")
        targets = targets.to("cpu")
        si_snr = self._compute_temporal_loss(predictions, targets)
        mag_loss, phase_loss = self._compute_spectral_losses(predictions, targets, alpha=0.5)
        loss = self._lambda_temp * (1 - si_snr / 100) + self._lambda_mag * mag_loss + self._lambda_phase * phase_loss
        return loss.to(device)

    def _compute_temporal_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return scale_invariant_signal_noise_ratio(predictions, targets).mean()

    def _compute_spectral_losses(
        self, predictions: Tensor, targets: Tensor, n_fft: int = 128, alpha: float = 1
    ) -> Tuple[Tensor, Tensor]:
        batch, _, time = predictions.shape
        epsilon = 1e-6
        stft_pred = stft(
            predictions.view(batch, time),
            n_fft=n_fft,
            window=hann_window(n_fft).to(targets.device),
            normalized=True,
            return_complex=True,
        )
        stft_target = stft(
            targets.view(batch, time),
            n_fft=n_fft,
            window=hann_window(n_fft).to(targets.device),
            normalized=True,
            return_complex=True,
        )

        stft_pred_mag = stft_pred.abs().clamp_min(epsilon)
        stft_target_mag = stft_target.abs().clamp_min(epsilon)
        mag_loss = mse_loss(stft_pred_mag**alpha, stft_target_mag**alpha)

        pred_phase = stft_pred.angle()
        target_phase = stft_target.angle()
        phase_diff = atan2(sin(target_phase - pred_phase), cos(target_phase - pred_phase))
        phase_loss = mse_loss(phase_diff, zeros_like(phase_diff))

        return mag_loss, phase_loss
