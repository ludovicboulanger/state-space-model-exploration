from abc import ABC
from typing import Any, Dict, Tuple

from lightning import LightningModule
from torch import Tensor
from torch.nn import Linear, ModuleList, Sequential
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import TrainingConfig
from models.s4 import S4Block
from models.ssm_encoder import SSMEncoder
from utils.encoders import Decimator


class SequenceModule(LightningModule, ABC):
    def __init__(self, config: TrainingConfig, output_dim: int) -> None:
        super().__init__()
        self._config = config
        self._inference_mode = False

        self._encoder = self._build_encoder()
        self._blocks = self._instantiate_isotropic_blocks()
        self._decoder = Linear(in_features=config.channel_dim, out_features=output_dim)

    def update_sampling_rate(self, factor: float) -> None:
        for layer in self._encoder:
            if isinstance(layer, SSMEncoder):
                layer.recompute_impulse_response(factor)

        for block in self._blocks:
            block.kernel.scale_factor = factor  # type: ignore

    def _instantiate_isotropic_blocks(self) -> ModuleList:
        residual_at_first_block = self._config.encoder != "dense"
        blocks = ModuleList([])
        for i in range(self._config.num_layers):
            blocks.append(
                S4Block(
                    in_channels=self._config.channel_dim,
                    out_channels=self._config.channel_dim,
                    n_ssms=self._config.num_ssms,
                    state_dim=self._config.hidden_dim,
                    min_dt=self._config.min_dt,
                    max_dt=self._config.max_dt,
                    clip_B=self._config.clip_B,
                    residual=residual_at_first_block if i == 0 else True,
                    p_kernel_dropout=self._config.kernel_dropout_prob,
                    p_block_dropout=self._config.block_dropout_prob,
                    norm=self._config.norm if i > 0 or residual_at_first_block else "none",
                    prenorm=self._config.pre_norm,
                    layer_activation=self._config.layer_activation,
                    final_activation=self._config.final_activation,
                )
            )
        return blocks

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore
        hippo_parameters = []
        nn_parameters = []
        for name, param in self.named_parameters():
            param_name = name.split(".")[-1]
            if param_name in ["A_real", "A_imag", "_P", "_B", "_C", "_dt"]:
                hippo_parameters.append(param)
            else:
                nn_parameters.append(param)

        optimizer = AdamW(
            params=[
                {"params": hippo_parameters, "lr": min(self._config.lr, 1e-3), "weight_decay": 0.0},
                {"params": nn_parameters, "lr": self._config.lr, "weight_decay": self._config.weight_decay},
            ],
        )
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=100_000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _build_encoder(self) -> Sequential:
        if self._config.encoder in ["s4", "lmu", "fout"]:
            """
                S4(
                    channels=1,
                    n_ssms=1,
                    state_dim=512,
                    min_dt=1e-1,
                    max_dt=1e-1,
                    output_state=True,
                ),

            """
            encoder = Sequential(
                SSMEncoder(
                    mode=self._config.encoder,
                    state_dim=512,
                    output_dim=self._config.channel_dim,
                    memory_size_seconds=self._config.encoder_memory_size_seconds,
                    seq_len=self._config.seq_len,
                ),
                Decimator(16_000, self._config.decimation_factor, antialias=False),
            )
        else:
            if self._config.decimation_factor > 1:
                encoder = Sequential(
                    Decimator(fs=16_000, decimation_factor=self._config.decimation_factor, antialias=True),
                    Linear(1, self._config.channel_dim),
                )
            else:
                encoder = Sequential(Linear(in_features=1, out_features=self._config.channel_dim))
        return encoder

    def _forward_pass(self, batch: Tuple[Tensor, ...], pool: bool) -> Tensor:
        return self._forward_isotropic(batch, pool)

    def _forward_isotropic(self, batch: Tuple[Tensor, ...], pool: bool) -> Tensor:
        x = batch[0]
        layer_outputs = []

        y_hat = self._encoder(x)
        for block in self._blocks:
            y_hat, layer_out = block(y_hat)
            layer_outputs.append(layer_out)
        if pool:
            y_hat = y_hat.mean(dim=1)
        logits = self._decoder(y_hat)
        return logits
