import torch
from module.Sersa_module import extract_adapter_params_for_experts

class ExpertEncodingCache:
    def __init__(self, expert_encoder):
        self.expert_encoder = expert_encoder

    def ensure_layer_encoded(self, model, layer_idx):
        adapter_map = extract_adapter_params_for_experts(model, layers_to_use=[layer_idx])
        if layer_idx not in adapter_map:
            raise ValueError(f"No LoRA params found in layer {layer_idx}.")
            
        expert_params = adapter_map[layer_idx].to(next(self.expert_encoder.parameters()).device).float()
        with torch.no_grad():
            encoded_experts = self.expert_encoder(expert_params)
            
        # keys 通常对应专家的索引 ID
        keys = list(range(encoded_experts.size(0)))
        return keys, encoded_experts.cpu()