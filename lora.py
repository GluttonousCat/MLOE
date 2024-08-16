import torch
import torch.nn as nn

from config import LoraConfig
from typing import Tuple


class LoraLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        config: LoraConfig,
        weight: Tuple[torch.Tensor, torch.Tensor] = (None, None),
        device: str = None,
    ):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise ValueError("Base layer must be of type nn.Linear.")
        out_dim, in_dim = base_layer.weight.shape

        self._base_layer_ = base_layer
        self._device = torch.device(device) if device else base_layer.weight.device
        self._dtype = config.dtype

        self._initializer = config.lora_init
        self._r = config.lora_r
        self._alpha = config.lora_alpha
        self._scaling = self._alpha / self._r

        self._in_features = in_dim
        self._out_features = out_dim

        assert config.lora_dropout > 0.0
        self._dropout = nn.Dropout(p=config.lora_dropout)

        self.lora_A = nn.Linear(
            self._in_features,
            self._r,
            bias=False,
            dtype=self._dtype,
            device=self._device,
        )
        self.lora_B = nn.Linear(
            self._r,
            self._out_features,
            bias=False,
            dtype=self._dtype,
            device=self._device,
        )

        self.reset_parameters(weight)

    def forward(self, x):

        original_output = self.base_layer(x)
        lora_output = self.lora_B(self.lora_A(x))

        return original_output + self._dropout(lora_output) * self._scaling
