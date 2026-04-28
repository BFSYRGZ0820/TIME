import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import json
import math
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import gzip
import json
from module.moe_lora import inject_lora_into_moe
from module.Sersa_module import extract_adapter_params_for_experts

class LocalC4Dataset(torch.utils.data.IterableDataset):
    def __init__(self, file_path, sample_tokenizer, moe_tokenizer, max_length=512, max_samples=50000):
        super().__init__()
        self.file_path = file_path
        self.sample_tokenizer = sample_tokenizer
        self.moe_tokenizer = moe_tokenizer
        self.max_length = max_length
        self.max_samples = max_samples

    def __iter__(self):
        count = 0
        # 直接流式读取本地的 .gz 压缩包，不占用内存
        with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if count >= self.max_samples:
                    break
                
                # 解析 JSONL 的每一行
                item = json.loads(line)
                text = item.get('text', '')
                
                if len(text.strip()) < 50:
                    continue

                sample_inputs = self.sample_tokenizer(
                    text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors='pt'
                )
                moe_inputs = self.moe_tokenizer(
                    text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors='pt'
                )

                yield {
                    "sample_input_ids": sample_inputs["input_ids"][0],
                    "sample_attention_mask": sample_inputs["attention_mask"][0],
                    "moe_input_ids": moe_inputs["input_ids"][0],
                    "moe_attention_mask": moe_inputs["attention_mask"][0]
                }
                count += 1

# ==========================================
# 1. 定义编码器模型 (Encoders)
# ==========================================
class SampleEncoder(nn.Module):
    def __init__(self, model_name='all-MiniLM-L6-v2', embedding_dim=256):
        super().__init__()
        # 论文中使用 all-MiniLM-L6-v2 作为文本特征提取的主干
        self.encoder = AutoModel.from_pretrained(model_name)
        # 将特征投影到对齐空间的维度 (256)
        self.projector = nn.Linear(self.encoder.config.hidden_size, embedding_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean Pooling 获取句子向量
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # 投影并使用 L2 归一化 (论文设定)
        embs = self.projector(pooled)
        return F.normalize(embs, p=2, dim=-1)

class ExpertEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=256):
        super().__init__()
        # 论文设定：两层 MLP + GELU 激活函数
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, expert_params):
        # expert_params: (M, input_dim), M 是专家数量 (Qwen1.5-MoE 为 60)
        embs = self.mlp(expert_params)
        return F.normalize(embs, p=2, dim=-1) # L2 归一化

# ==========================================
# 2. 双向对比学习损失函数
# ==========================================
def bidirectional_contrastive_loss(sample_embs, expert_embs, positive_masks, tau=0.07):
    """
    sample_embs: (Batch_size, 256)
    expert_embs: (M, 256) - 某一层的所有专家
    positive_masks: (Batch_size, M) - 布尔矩阵，True 表示匹配
    tau: 温度超参数 (论文设为 0.07)
    """
    # 计算欧氏距离 (Euclidean Distance)
    dists = torch.cdist(sample_embs, expert_embs, p=2.0)
    sims = -dists / tau  # 距离越小，相似度越大

    # --- L_{s->e}: 样本到专家的对比损失 ---
    sims_pos_s2e = sims.clone()
    sims_pos_s2e[~positive_masks] = -float('inf')
    
    log_num_s2e = torch.logsumexp(sims_pos_s2e, dim=1) # 分子: 正样本的 sum(exp)
    log_den_s2e = torch.logsumexp(sims, dim=1)         # 分母: 所有专家的 sum(exp)
    loss_s2e = -(log_num_s2e - log_den_s2e).mean()

    # --- L_{e->s}: 专家到样本的反向对比损失 ---
    sims_e2s = sims.t()  # (M, Batch_size)
    masks_e2s = positive_masks.t()
    
    sims_pos_e2s = sims_e2s.clone()
    sims_pos_e2s[~masks_e2s] = -float('inf')
    
    # 仅计算在当前 Batch 中被当作正样本的专家
    valid_experts = masks_e2s.sum(dim=1) > 0
    if valid_experts.sum() > 0:
        log_num_e2s = torch.logsumexp(sims_pos_e2s[valid_experts], dim=1)
        log_den_e2s = torch.logsumexp(sims_e2s[valid_experts], dim=1)
        loss_e2s = -(log_num_e2s - log_den_e2s).mean()
    else:
        loss_e2s = 0.0

    return loss_s2e + loss_e2s

# ==========================================
# 3. 正负样本划分策略
# ==========================================
def get_positive_experts(moe_model, input_ids, strategy="or", top_k=2, layer_idx=0):
    """
    获取每个样本的正样本专家 ID 列表。
    """
    moe_model.eval()
    with torch.no_grad():
        if strategy == "or":
            # Original Routing (OR) 策略：直接使用预训练 Router 的输出
            outputs = moe_model(input_ids=input_ids, output_router_logits=True)
            # 获取目标层的 router logits，形状为 (Batch * Seq_len, Num_experts)
            router_logits = outputs.router_logits[layer_idx] 
            # 获取真实的 Batch size 和 Sequence length
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            num_experts = router_logits.size(-1)
            # 将其还原为 (Batch, Seq_len, Num_experts)
            router_logits = router_logits.view(batch_size, seq_len, num_experts)
            # 现在对 Sequence 维度 (dim=1) 取平均，得到真正的每个样本对专家的偏好 (Batch, Num_experts)
            mean_logits = router_logits.mean(dim=1)
            # 获取 Top-K 被激活的专家，取值范围必然在 0~59 之间
            _, top_experts = torch.topk(mean_logits, k=top_k, dim=-1)
            return top_experts
            
        elif strategy == "eoe":
            # Expert Output Entropy (EOE) 策略：寻找预测熵最低的专家
            # 注意：严谨的 EOE 需要对每个专家单独 forward，计算量极大。
            # 这里提供计算逻辑框架：实际应用中需要获取每个专家的 logits。
            # 假设我们能获取到每个专家独立输出的 logits -> (Batch, Num_experts, Vocab_size)
            # pseudo_expert_logits = get_all_expert_logits(moe_model, input_ids, layer_idx)
            # probs = torch.softmax(pseudo_expert_logits, dim=-1)
            # entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1) # (Batch, Num_experts)
            # _, top_experts = torch.topk(-entropy, k=top_k, dim=-1) # 熵越小越好 (加负号取 topk)
            # return top_experts
            
            raise NotImplementedError("EOE策略需要侵入模型底层获取每个专家的独立预测，建议优先使用 'or' 策略。")

# ==========================================
# 4. 主训练逻辑
# ==========================================
def train_ssa(moe_model_path, minilm_path, output_path, strategy="or", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    # 1. 加载模型与分词器
    moe_tokenizer = AutoTokenizer.from_pretrained(moe_model_path)
    moe_model = AutoModelForCausalLM.from_pretrained(moe_model_path, device_map="auto", torch_dtype=torch.bfloat16)
    sample_tokenizer = AutoTokenizer.from_pretrained(minilm_path)
    
    # 2. 注入 LoRA 并提取真实的专家参数
    moe_model = inject_lora_into_moe(moe_model, r=16, alpha=16)
    adapter_map = extract_adapter_params_for_experts(moe_model, layers_to_use=[0])
    expert_params = adapter_map[0].to(device).float() # 自动获取真实的形状 (60, 165888)
    lora_param_dim = expert_params.shape[1]
    num_experts = expert_params.shape[0]
    print(f"Extracted real LoRA params with shape: {expert_params.shape}")
    
    # 3. 初始化编码器与优化器
    f_s = SampleEncoder(model_name=minilm_path).to(device)
    f_e = ExpertEncoder(input_dim=lora_param_dim).to(device)
    optimizer = torch.optim.Adam(list(f_s.parameters()) + list(f_e.parameters()), lr=1e-4)
    
    # ---------------------------------------------------------
    # 4. 初始化真实的 C4 流式 DataLoader
    # ---------------------------------------------------------
    print("Initializing Streaming C4 Dataset...")
    # 设置 max_samples 决定你希望用来训练的样本总量 (论文中离线阶段不需要跑全量数据)
    c4_dataset = LocalC4Dataset(
        file_path="/wenzhiquan/mr/datasets/c4/c4-train.00000-of-01024.json.gz", # 只用c4一个切片（大概 300MB），包含了约 30 万条文本
        sample_tokenizer=sample_tokenizer, 
        moe_tokenizer=moe_tokenizer, 
        max_length=512, 
        max_samples=200000 
    )
    # 注意：IterableDataset 不要设置 shuffle，也不要设置过多的 num_workers，直接用单进程即可
    dataloader = DataLoader(c4_dataset, batch_size=batch_size)
    
    # 5. 训练循环
    f_s.train()
    f_e.train()
    
    print(f"Starting offline training using strategy: {strategy.upper()}")
    
    step = 0
    # 由于是 IterableDataset，我们直接遍历 dataloader 即可
    for batch in dataloader:
        # 将数据移至 GPU
        sample_input_ids = batch["sample_input_ids"].to(device)
        sample_attention_mask = batch["sample_attention_mask"].to(device)
        moe_input_ids = batch["moe_input_ids"].to(device)
        
        current_batch_size = sample_input_ids.size(0)
        
        # A. 获取样本的正样本专家 ID
        top_experts = get_positive_experts(moe_model, moe_input_ids, strategy=strategy)
        
        # 构建 Positive Mask
        positive_masks = torch.zeros(current_batch_size, num_experts, dtype=torch.bool, device=device)
        for b in range(current_batch_size):
            positive_masks[b, top_experts[b]] = True
            
        # B. 前向计算：编码样本与专家
        s_embs = f_s(sample_input_ids, sample_attention_mask) 
        e_embs = f_e(expert_params)
        
        # C. 计算对比损失并反向传播
        loss = bidirectional_contrastive_loss(s_embs, e_embs, positive_masks, tau=0.07)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
        step += 1
        
    # 6. 保存权重
    torch.save({
        'sample_encoder_state': f_s.state_dict(),
        'expert_encoder_state': f_e.state_dict()
    }, output_path)
    print(f"Encoders saved to {output_path}")

if __name__ == "__main__":
    # 请根据你的实际路径修改以下变量
    train_ssa(
        moe_model_path="./Qwen1.5-MoE-A2.7B-Chat",
        minilm_path="./all-MiniLM-L6-v2",
        output_path="./Sersa_outputs/final.pt",
        strategy="or"
    )