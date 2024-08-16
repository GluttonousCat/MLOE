import copy
from dataclasses import dataclass
from typing import Dict, List

import torch

lora_target_modules = {
    "q_proj": False,
    "k_proj": False,
    "v_proj": False,
    "o_proj": False,
    "gate_proj": False,
    "down_proj": False,
    "up_proj": False,
}


@dataclass
class AdapterConfig:
    _base_model: str = None
    _task_type: str = None
    _peft_type: str = None
    _adapter_name: str = None
    _model_type: str = None
    _dtype: torch.dtype = None

    @property
    def base_model_name_or_path(self):
        return self._base_model

    @property
    def task_type(self):
        return self._task_type

    @property
    def peft_type(self):
        return self.peft_type

    @property
    def adapter_name(self):
        return self._adapter_name

    @property
    def model_type(self):
        return self._model_type

    @property
    def dtype(self):
        return self._dtype

    @staticmethod
    def from_config(config: Dict[str, any]) -> "AdapterConfig":
        return AdapterConfig(
            _base_model=config.get("base_model"),
            _task_type=config.get("task_type"),
            _peft_type=config.get("peft_type"),
            _adapter_name=config.get("adapter_name"),
            _model_type=config.get("model_type"),
            _dtype=config.get("dtype")
        )

    def export(self) -> Dict[str, any]:
        config = {"bias": "none", "peft_type": self.peft_type, "task_type": self.task_type,
                  "base_model_name_or_path": self.base_model_name_or_path}

        return config


@dataclass
class LoraConfig(AdapterConfig):

    lora_init: str = "original"
    lora_r: int = None
    lora_alpha: int = None
    lora_dropout: float = None
    target_modules: Dict[str, bool] = None

    @staticmethod
    def from_config(config: Dict[str, any]) -> "LoraConfig":
        lora_config = LoraConfig(**AdapterConfig.from_config(config).__dict__)
        lora_config.lora_init = config.get("lora_init", "original")
        lora_config.lora_r = config["r"]
        lora_config.lora_alpha = config["lora_alpha"]
        lora_config.lora_dropout = config["lora_dropout"]
        lora_config.target_modules = copy.deepcopy(lora_target_modules)
        if isinstance(config["target_modules"], List):
            for target in config["target_modules"]:
                if target in lora_target_modules:
                    lora_config.target_modules[target] = True
        elif isinstance(config["target_modules"], Dict):
            for target, value in config["target_modules"].items():
                if target in lora_target_modules:
                    lora_config.target_modules[target] = value
        else:
            raise ValueError("broken config item: target_modules")

        return lora_config

    def export(self) -> Dict[str, any]:
        config = super().export()

        config["r"] = self.lora_r
        config["lora_alpha"] = self.lora_alpha
        config["lora_dropout"] = self.lora_dropout
        tgt_list = []
        for target, value in self.target_modules.items():
            if value:
                tgt_list.append(target)
        config["target_modules"] = tgt_list

        return config


class MixLoraMoeConfig(LoraConfig):
    _act_fn: str = None
    _num_experts: int = None
    _top_k: int = None
    _jitter_noise: float = None

    @property
    def act_fn(self):
        return self._act_fn

    @property
    def num_experts(self):
        return self._num_experts

    @property
    def top_k(self):
        return self._top_k

    @property
    def jitter_noise(self):
        return self._jitter_noise

