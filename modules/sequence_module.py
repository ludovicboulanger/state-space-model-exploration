from abc import ABC
from math import ceil, floor
from typing import Any, Dict, Tuple

from lightning import LightningModule
from torch import Tensor
from torch.nn import Linear, ModuleList
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import TrainingConfig
from models.s4 import S4Block


class SequenceModule(LightningModule, ABC):
    def __init__(self, config: TrainingConfig, output_dim: int) -> None:
        super().__init__()
        self._config = config
        self._inference_mode = False

        self._encoder = Linear(in_features=1, out_features=config.channel_dim)

        if config.unet:
            self._blocks = self._instantiate_unet_blocks()
        else:
            self._blocks = self._instantiate_regular_blocks()
        self._decoder = Linear(in_features=config.channel_dim, out_features=output_dim)

    def _instantiate_regular_blocks(self) -> ModuleList:
        blocks = ModuleList([])
        for i in range(self._config.num_layers):
            blocks.append(
                S4Block(
                    in_channels=self._config.channel_dim,
                    out_channels=self._config.channel_dim,
                    n_ssms=self._config.num_ssms,
                    state_dim=self._config.hidden_dim,
                    seq_len=self._config.seq_len,
                    min_dt=self._config.min_dt,
                    max_dt=self._config.max_dt,
                    clip_B=self._config.clip_B,
                    residual=i > 0,
                    p_kernel_dropout=self._config.kernel_dropout_prob,
                    p_block_dropout=self._config.block_dropout_prob,
                    norm=self._config.norm if i > 0 else "none",
                    prenorm=self._config.pre_norm,
                    layer_activation=self._config.layer_activation,
                    final_activation=self._config.final_activation,
                )
            )
        return blocks

    def _instantiate_unet_blocks(self) -> ModuleList:
        if self._config.num_ssms > -1:
            min_channels = self._config.num_ssms
        else:
            min_channels = 1
        num_encoder_layers = self._config.num_layers // 2
        reduction_factor = floor((self._config.channel_dim - min_channels) ** (1 / num_encoder_layers))

        blocks = ModuleList([])
        out_channels = self._config.channel_dim
        for i in range(num_encoder_layers):
            blocks.append(
                S4Block(
                    in_channels=out_channels,
                    out_channels=max(out_channels // reduction_factor, min_channels),
                    n_ssms=self._config.num_ssms,
                    state_dim=self._config.hidden_dim,
                    seq_len=self._config.seq_len,
                    min_dt=self._config.min_dt,
                    max_dt=self._config.max_dt,
                    clip_B=self._config.clip_B,
                    residual=i > 0,
                    p_kernel_dropout=self._config.kernel_dropout_prob,
                    p_block_dropout=self._config.block_dropout_prob,
                    norm=self._config.norm if i > 0 else "none",
                    prenorm=self._config.pre_norm,
                    layer_activation=self._config.layer_activation,
                    final_activation=self._config.final_activation,
                )
            )
            if i != num_encoder_layers - 1:
                out_channels = out_channels // reduction_factor

        in_channels = max(out_channels // reduction_factor, min_channels)
        upsampling_factor = ceil((self._config.channel_dim - min_channels) ** (1 / num_encoder_layers))
        for i in range(num_encoder_layers):
            blocks.append(
                S4Block(
                    in_channels=in_channels,
                    out_channels=min(int(in_channels * upsampling_factor), self._config.channel_dim),
                    n_ssms=self._config.num_ssms,
                    state_dim=self._config.hidden_dim,
                    seq_len=self._config.seq_len,
                    min_dt=self._config.min_dt,
                    max_dt=self._config.max_dt,
                    clip_B=self._config.clip_B,
                    residual=True,
                    p_kernel_dropout=self._config.kernel_dropout_prob,
                    p_block_dropout=self._config.block_dropout_prob,
                    norm=self._config.norm if i > 0 else "none",
                    prenorm=self._config.pre_norm,
                    layer_activation=self._config.layer_activation,
                    final_activation=self._config.final_activation,
                )
            )
            in_channels = int(in_channels * upsampling_factor)
        return blocks

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore
        hippo_parameters = []
        nn_parameters = []
        for name, param in self._blocks.named_parameters():
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
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=self._config.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def update_sampling_rate(self, sampling_factor: int) -> None:
        for block in self._blocks:
            block._s4_layer._kernel.scale_factor = sampling_factor  # type: ignore

    def _forward_pass(self, batch: Tuple[Tensor, ...], pool: bool) -> Tensor:
        x, y, _ = batch
        layer_outputs = []

        y_hat = self._encoder(x)
        for block in self._blocks:
            y_hat, layer_out = block(y_hat)
            layer_outputs.append(layer_out)
        if pool:
            y_hat = y_hat.mean(dim=1)
        logits = self._decoder(y_hat)
        return logits
