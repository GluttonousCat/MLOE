from dataclasses import dataclass
from typing import Tuple, Dict

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.activations import ACT2FN

from config import LoraConfig, MixLoraMoeConfig
from lora import LoraLinear


class MixLoraMoe(torch.nn.Module):
    def __init__(
        self,
        base_layer: torch.nn.Module,
        config: MixLoraMoeConfig,
    ) -> None:
        super().__init__()
        self.dtype: torch.dtype = config.dtype
        self.gate: torch.Tensor
        self.base_layer: torch.nn.Module = base_layer
        self.experts: Dict[str, LoraLinear] = {}
        self.act_fn = (
            ACT2FN[config.act_fn]
            if isinstance(config.act_fn, str)
            else config.act_fn
        )
        self.num_experts_: int = config.num_experts
        self.top_k: int = config.top_k
        self.jitter_noise_: float = config.jitter_noise

        self.forward_fn_ = getattr(self, "_llama_forward")

        self._add_lora_module()

    def _add_lora_module(self):
        for layer in self.base_layer.model.layers:
            for proj_name, inject in layer.items():
                if not inject or not hasattr(layer, proj_name):
                    continue
            base_layer = getattr(layer, proj_name)
            lora_layer_prefix_name = f"layers.{layer_idx}.self_attn.{proj_name}"
            setattr(
                layer,
                proj_name,
                LoraLinear(
                    base_layer,
                    config,
                    (
                        weights[f"{lora_layer_prefix_name}.lora_A.weight"],
                        weights[f"{lora_layer_prefix_name}.lora_B.weight"],
                    ),
                ),

            )


    def _llama_forward(
            self,
            expert_mask: torch.Tensor,
            hidden_states: torch.Tensor,
            input_dtype: torch.dtype,
    ):
        pass



def _inject_lora_module(
        layer_idx: int,
        layer: torch.nn.Module,
        config: LoraConfig,
        weights: Dict[str, torch.Tensor],
):
    for proj_name, inject in config.target_modules.items():
        if not inject or not hasattr(layer, proj_name):
            continue
        base_layer = getattr(layer, proj_name)
        lora_layer_prefix_name = f"layers.{layer_idx}.self_attn.{proj_name}"
        setattr(
            layer,
            proj_name,
            LoraLinear(
                base_layer,
                config,
                (
                    weights[f"{lora_layer_prefix_name}.lora_A.weight"],
                    weights[f"{lora_layer_prefix_name}.lora_B.weight"],
                ),
            ),
        )


def _inject_adapter_in_model(
    model: PreTrainedModel,
    config: LoraConfig,
    weights: Dict[str, torch.Tensor],
):
    config.model_type_ = model.config.model_type
    model._mixlora_config = config
    for idx, layer in enumerate(model.model.layers):
        _inject_lora_module(idx, layer, config, weights)


@dataclass
class MixLoraModelForCausalLM:
    @staticmethod
    def from_pretrained(
        name_or_path: str,
        *model_args,
        **kwargs,
    ) -> Tuple[PreTrainedModel, LoraConfig]:
        config, weights = load_adapter_weights(
            name_or_path,
            adapter_name=kwargs.pop("adapter_name", "default"),
            dtype=kwargs.get("torch_dtype", torch.float32),
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_, *model_args, **kwargs
        )

        _inject_adapter_in_model(model, config, weights)

        return model, config
