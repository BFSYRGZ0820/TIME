import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class TrainableSampleEncoder(nn.Module):
    def __init__(self, encoder_name='./all-MiniLM-L6-v2', embedding_dim=256, device="cpu"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.projector = nn.Linear(self.encoder.config.hidden_size, embedding_dim)
        self.to(device)

    def forward(self, texts):
        # 接收文本列表，并在内部进行分词，因为 CTTA 主程序传入的是纯文本列表
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        
        outputs = self.encoder(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        
        # Mean Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # 投影与 L2 归一化
        embs = self.projector(pooled)
        return F.normalize(embs, p=2, dim=-1)

class ExpertEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=256, device="cpu"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, embedding_dim)
        )
        self.to(device)

    def forward(self, expert_params):
        embs = self.mlp(expert_params)
        return F.normalize(embs, p=2, dim=-1)



def extract_adapter_params_for_experts(model, layers_to_use=[0]):
    """提取指定 MoE 层中所有专家的 LoRA 展平参数 (支持所有 transformers 版本)"""
    adapter_map = {}
    
    for layer_idx in layers_to_use:
        expert_lora_params = {}
        vectorized_tensors = []
        is_vectorized = False
        
        # 1. 匹配老版本独立专家: ...experts.14.down_proj.lora...
        pattern_ind = re.compile(r"layers\.{}\..*?experts\.(\d+)\..*lora".format(layer_idx), re.IGNORECASE)
        # 2. 匹配新版本向量化专家: ...experts.down_proj.lora...
        pattern_vec = re.compile(r"layers\.{}\..*?experts\.(?:down_proj|up_proj|gate_proj).*lora".format(layer_idx), re.IGNORECASE)
        
        for name, param in model.named_parameters():
            if "lora" in name.lower() and f"layers.{layer_idx}." in name:
                # 尝试匹配独立专家
                match_ind = pattern_ind.search(name)
                if match_ind:
                    expert_idx = int(match_ind.group(1))
                    if expert_idx not in expert_lora_params:
                        expert_lora_params[expert_idx] = []
                    expert_lora_params[expert_idx].append(param.detach().view(-1))
                    continue
                
                # 尝试匹配向量化专家
                match_vec = pattern_vec.search(name)
                if match_vec:
                    is_vectorized = True
                    vectorized_tensors.append(param.detach())
        
        # --- 组装逻辑 ---
        if not is_vectorized and expert_lora_params:
            num_experts = max(expert_lora_params.keys()) + 1
            layer_params = []
            for i in range(num_experts):
                if i in expert_lora_params and len(expert_lora_params[i]) > 0:
                    layer_params.append(torch.cat(expert_lora_params[i]))
            if layer_params:
                adapter_map[layer_idx] = torch.stack(layer_params)
                
        elif is_vectorized and vectorized_tensors:
            try:
                # 假设张量的第 0 维是专家数量 (例如 60)
                num_experts = vectorized_tensors[0].shape[0]
                layer_params = []
                for i in range(num_experts):
                    expert_parts = [tensor[i].contiguous().view(-1) for tensor in vectorized_tensors]
                    layer_params.append(torch.cat(expert_parts))
                adapter_map[layer_idx] = torch.stack(layer_params)
            except Exception as e:
                pass

        # --- 诊断与自愈防崩溃 ---
        if layer_idx not in adapter_map:
            print("\n" + "!"*60)
            print(f"[警告] 第 {layer_idx} 层没有找到任何路由专家的 LoRA 参数！")
            print("PEFT 实际挂载的 LoRA 参数列表如下，请检查是否全是 shared_expert：")
            for name, _ in model.named_parameters():
                if "lora" in name.lower() and f"layers.{layer_idx}." in name:
                    print(" ->", name)
            
            print("\n正在插入占位 Dummy 张量以防止代码崩溃 (CTTA效果将受影响)...")
            # 为 Qwen1.5-MoE 插入 (60, 131072) 的占位符
            adapter_map[layer_idx] = torch.zeros(60, 131072)
            print("!"*60 + "\n")

    return adapter_map