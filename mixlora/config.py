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
    base_model_: str = None
    task_type_: str = None
    peft_type_: str = None
    adapter_name_: str = None
    model_type_: str = None
    dtype_: torch.dtype = None

    @property
    def base_model_name_or_path(self):
        return self.base_model_

    @property
    def adapter_name(self):
        return self.adapter_name_

    @staticmethod
    def from_config(config: Dict[str, any]) -> "AdapterConfig":
        return AdapterConfig(
            base_model_=config["base_model_name_or_path"],
            task_type_=config["task_type"],
            peft_type_=config["peft_type"],
        )

    def export(self) -> Dict[str, any]:
        config = {"bias": "none", "peft_type": self.peft_type_, "task_type": self.task_type_,
                  "base_model_name_or_path": self.base_model_}

        return config


@dataclass
class LoraConfig(AdapterConfig):

    lora_init_: str = "original"
    lora_r_: int = None
    lora_alpha_: int = None
    lora_dropout_: float = None
    target_modules_: Dict[str, bool] = None

    @staticmethod
    def from_config(config: Dict[str, any]) -> "LoraConfig":
        lora_config = LoraConfig(**AdapterConfig.from_config(config).__dict__)
        lora_config.lora_init_ = config.get("lora_init", "original")
        lora_config.lora_r_ = config["r"]
        lora_config.lora_alpha_ = config["lora_alpha"]
        lora_config.lora_dropout_ = config["lora_dropout"]
        lora_config.target_modules_ = copy.deepcopy(lora_target_modules)
        if isinstance(config["target_modules"], List):
            for target in config["target_modules"]:
                if target in lora_target_modules:
                    lora_config.target_modules_[target] = True
        elif isinstance(config["target_modules"], Dict):
            for target, value in config["target_modules"].items():
                if target in lora_target_modules:
                    lora_config.target_modules_[target] = value
        else:
            raise ValueError("broken config item: target_modules")

        return lora_config

    def export(self) -> Dict[str, any]:
        config = super().export()

        config["r"] = self.lora_r_
        config["lora_alpha"] = self.lora_alpha_
        config["lora_dropout"] = self.lora_dropout_
        tgt_list = []
        for target, value in self.target_modules_.items():
            if value:
                tgt_list.append(target)
        config["target_modules"] = tgt_list

        return config
