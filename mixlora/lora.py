import torch
import torch.nn as nn

from .config import LoraConfig


class LoraLinear(nn.Module):
    def __init__(self, base_layer: nn.Module, config: LoraConfig, device: str = None):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError('Base layer must be a nn.Linear module')

        out_dim, in_dim = base_layer.weight.shape

        self.base_layer_ = base_layer
        self.device_ = torch.device(device) if device else base_layer.weight.device
        self.dtype_ = config.dtype_

        self.initializer_ = config.lora_init_
        self.r_ = config.lora_r_
        self.alpha_ = config.lora_alpha_

        self.scaling_ = self.alpha_ / self.r_

        self.in_features_ = in_dim
        self.out_features_ = out_dim

        assert config.lora_dropout_ > 0.0
        self.dropout_ = nn.Dropout(p=config.lora_dropout_)

        self.lora_A = nn.Linear(
            self.in_features_,
            self.r_,
            bias=False,
            dtype=self.dtype_,
            device=self.device_,
        )

        self.lora_B = nn.Linear(
            self.r_,
            self.out_features_,
            bias=False,
            dtype=self.dtype_,
            device=self.device_,
        )




