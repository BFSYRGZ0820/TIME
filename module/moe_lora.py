import re

import torch
from peft import LoraConfig, get_peft_model


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _get_layer_index(name: str):
    match = _LAYER_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def _is_lora_param(name: str) -> bool:
    return "lora" in name.lower()


def _is_shared_expert_name(name: str) -> bool:
    name_lower = name.lower()
    return "shared_expert" in name_lower or "shared_experts" in name_lower


def _is_routed_expert_lora_name(name: str) -> bool:
    if not _is_lora_param(name):
        return False
    name_lower = name.lower()
    if _is_shared_expert_name(name_lower):
        return False
    return ".experts." in name_lower


def _find_routed_expert_lora_layers(model):
    layers = set()
    for name, _ in model.named_parameters():
        if _is_routed_expert_lora_name(name):
            layer_idx = _get_layer_index(name)
            if layer_idx is not None:
                layers.add(layer_idx)
    return sorted(layers)


def _find_trainable_target_layers(model, requested_layers=None):
    routed_layers = _find_routed_expert_lora_layers(model)
    if requested_layers is None:
        if routed_layers:
            return [routed_layers[0]]
        return []

    requested = list(requested_layers)
    if not routed_layers:
        return []

    selected = [layer_idx for layer_idx in requested if layer_idx in routed_layers]
    return selected


def inject_lora_into_moe(model, r=16, alpha=16):
    """向 MoE FFN 投影层注入 LoRA，后续再按 routed expert 进行筛选。"""
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["down_proj", "up_proj", "gate_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    return model


def freeze_non_lora_params(model, layers_to_use=None):
    """
    只保留指定 MoE 层 routed experts 的 LoRA 参数可训练。

    - 默认自动选择当前模型中“第一个真正含 routed expert LoRA 参数”的层。
    - 自动排除 shared expert。
    - 若未识别到 routed expert LoRA，则退化为保留全部 LoRA 参数，并打印警告。
    """
    selected_layers = _find_trainable_target_layers(model, requested_layers=layers_to_use)
    found_any_routed = len(_find_routed_expert_lora_layers(model)) > 0

    trainable_count = 0
    for name, param in model.named_parameters():
        keep_trainable = False
        if _is_lora_param(name):
            layer_idx = _get_layer_index(name)
            if selected_layers:
                keep_trainable = (
                    layer_idx in selected_layers and _is_routed_expert_lora_name(name)
                )
            elif not found_any_routed:
                keep_trainable = True

        param.requires_grad = keep_trainable
        if keep_trainable:
            trainable_count += 1

    if selected_layers:
        print(f"[moe_lora] Trainable routed-expert LoRA layers: {selected_layers}")
    elif found_any_routed:
        print(
            "[moe_lora][Warning] Requested layers do not contain routed expert LoRA params. "
            "All parameters were frozen."
        )
    else:
        print(
            "[moe_lora][Warning] No routed expert LoRA params were identified. "
            "Falling back to training all LoRA parameters."
        )

    print(f"[moe_lora] Trainable parameter tensors after freezing: {trainable_count}")


def register_layer_router_hook(model, routed_experts_dict, layers_to_track=[0], topk=2):
    """用于追踪哪些专家被原始 router 激活，这里作为占位接口以防报错"""
    pass
