"""T-IoI-UNR: A Flower / FlowerTune app."""

import math

import torch
from omegaconf import DictConfig
from collections import OrderedDict
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from flwr.common.typing import NDArrays

# from transformers.utils.logging import set_verbosity_error
# set_verbosity_error()


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.
    """
    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )
    if 'DeepSeek-Coder' in model_cfg.name:
        print('deepseek in my model')
        peft_config = LoraConfig(
            use_dora=True,
            r=model_cfg.lora.peft_lora_r,
            lora_alpha=model_cfg.lora.peft_lora_alpha,
            lora_dropout=0.075,
            task_type="CAUSAL_LM",
            target_modules = [
                "q_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
            ],
            # bias="none",
        )
    else:
        peft_config = LoraConfig(
            use_dora=True,
            r=model_cfg.lora.peft_lora_r,
            lora_alpha=model_cfg.lora.peft_lora_alpha,
            lora_dropout=0.075,
            task_type="CAUSAL_LM",
        )

    if model_cfg.gradient_checkpointing:
        model.config.use_cache = False
        # model.generation_config.cache_implementation = "none"  # set explicitly
        model.config.cache_implementation = "none"  # set explicitly

    return get_peft_model(model, peft_config)


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def get_parameters(model) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]
