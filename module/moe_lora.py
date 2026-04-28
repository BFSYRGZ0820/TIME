import torch
from peft import LoraConfig, get_peft_model

def inject_lora_into_moe(model, r=16, alpha=16):
    """仅向 MoE 层的专家网络注入 LoRA"""
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        # 针对 Qwen1.5-MoE 的专家模块进行注入
        target_modules=["down_proj", "up_proj", "gate_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    return model

'''
def freeze_non_lora_params(model):
    """严格限制只训练第 0 层的 lora 适配器（契合作者 para_initial 的限制）"""
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and '.layers.0.' in name:
            param.requires_grad = True 
        else:
            param.requires_grad = False       
'''

def freeze_non_lora_params(model):
    """严格对齐作者的正则化范围，保护共享专家(shared_expert)不发生崩盘"""
    for name, param in model.named_parameters():
        # 必须同时满足三个条件：是 lora、在第 0 层、并且是 routed experts（排除 shared_expert）
        if 'lora' in name.lower() and '.layers.0.' in name and '.mlp.experts.' in name:
            param.requires_grad = True 
        else:
            param.requires_grad = False


def register_layer_router_hook(model, routed_experts_dict, layers_to_track=[0], topk=2):
    """用于追踪哪些专家被原始 router 激活，这里作为占位接口以防报错"""
    pass