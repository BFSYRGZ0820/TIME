import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import gzip
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader

# 严格调用官方核心模块
from module.moe_lora import inject_lora_into_moe
from module.Sersa_module import extract_adapter_params_for_experts, TrainableSampleEncoder, ExpertEncoder

class AlignedC4Dataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512, max_samples=100000):
        self.texts = []
        print(f"Loading data from {path} into memory...")
        
        # 兼容原作者的 .json 或压缩包 .json.gz
        open_func = gzip.open if path.endswith('.gz') else open
        
        with open_func(path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                obj = json.loads(line)
                
                # [致敬原作者] 严格复刻作者的强制字符截断 [:20000]
                text = obj.get('text', '')[:20000]
                
                # 过滤掉几乎无意义的极短文本
                if len(text.strip()) < 50:
                    continue
                self.texts.append(text)
                
        print(f"Successfully loaded {len(self.texts)} samples into memory.")
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # MoE Router 需要的 Token 张量
        enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors='pt')
        return {
            'text': text, # 必须保留纯文本，传给 TrainableSampleEncoder
            'moe_input_ids': enc['input_ids'][0],
            'moe_attention_mask': enc['attention_mask'][0]
        }

# ==========================================
# 2. 核心补丁：带 Padding 屏蔽的 Router 偏好提取
# ==========================================
def get_positive_experts(moe_model, input_ids, attention_mask, top_k=2, layer_idx=0):
    moe_model.eval()
    with torch.no_grad():
        outputs = moe_model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=True)
        router_logits = outputs.router_logits[layer_idx] 
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        num_experts = router_logits.size(-1)
        
        router_logits = router_logits.view(batch_size, seq_len, num_experts)
        
        # 抹杀所有 <pad> 产生的 Logits，防止垃圾专家污染
        mask_expanded = attention_mask.unsqueeze(-1)
        router_logits_masked = router_logits * mask_expanded
        
        sum_logits = router_logits_masked.sum(dim=1)
        valid_tokens = attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_logits = sum_logits / valid_tokens
        
        _, top_experts = torch.topk(mean_logits, k=top_k, dim=-1)
        return top_experts

# ==========================================
# 3. 双向对比学习损失函数
# ==========================================
def bidirectional_contrastive_loss(sample_embs, expert_embs, positive_masks, tau=0.07):
    dists = torch.cdist(sample_embs, expert_embs, p=2.0)
    sims = -dists / tau  

    sims_pos_s2e = sims.clone()
    sims_pos_s2e[~positive_masks] = -float('inf')
    log_num_s2e = torch.logsumexp(sims_pos_s2e, dim=1) 
    log_den_s2e = torch.logsumexp(sims, dim=1)         
    loss_s2e = -(log_num_s2e - log_den_s2e).mean()

    sims_e2s = sims.t()  
    masks_e2s = positive_masks.t()
    sims_pos_e2s = sims_e2s.clone()
    sims_pos_e2s[~masks_e2s] = -float('inf')
    
    valid_experts = masks_e2s.sum(dim=1) > 0
    if valid_experts.sum() > 0:
        log_num_e2s = torch.logsumexp(sims_pos_e2s[valid_experts], dim=1)
        log_den_e2s = torch.logsumexp(sims_e2s[valid_experts], dim=1)
        loss_e2s = -(log_num_e2s - log_den_e2s).mean()
    else:
        loss_e2s = torch.tensor(0.0, device=sample_embs.device)

    return loss_s2e + loss_e2s

# ==========================================
# 4. 主训练逻辑
# ==========================================
def train_ssa(moe_model_path, minilm_path, c4_path, output_path, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    moe_tokenizer = AutoTokenizer.from_pretrained(moe_model_path, local_files_only=True)
    moe_model = AutoModelForCausalLM.from_pretrained(moe_model_path, device_map="auto", torch_dtype=torch.bfloat16, local_files_only=True)
    
    moe_model = inject_lora_into_moe(moe_model, r=16, alpha=16)
    adapter_map = extract_adapter_params_for_experts(moe_model, layers_to_use=[0])
    expert_params = adapter_map[0].to(device).float() 
    lora_param_dim = expert_params.shape[1]
    num_experts = expert_params.shape[0]
    
    f_s = TrainableSampleEncoder(encoder_name=minilm_path, embedding_dim=256, device=device)
    f_e = ExpertEncoder(input_dim=lora_param_dim, embedding_dim=256, device=device)
    
    optimizer = torch.optim.AdamW(list(f_s.parameters()) + list(f_e.parameters()), lr=1e-4)
    
    # 初始化作者原汁原味的本地数据集
    c4_dataset = AlignedC4Dataset(
        path=c4_path,
        tokenizer=moe_tokenizer, 
        max_len=512,
        max_samples=100000 # 读取 10 万条作为训练池
    )
    
    # 既然在内存里了，就可以开启 shuffle=True 来打破数据分布偏见！
    dataloader = DataLoader(c4_dataset, batch_size=batch_size, shuffle=True)
    
    f_s.train()
    f_e.train()
    
    step = 0
    max_steps = 5000 
    
    print("\nStarting Offline SSA Training...")
    
    for batch in dataloader:
        if step >= max_steps:
            break
            
        moe_input_ids = batch["moe_input_ids"].to(device)
        moe_attention_mask = batch["moe_attention_mask"].to(device)
        current_batch_size = moe_input_ids.size(0)
        
        # 核心：将 mask 传给偏好提取器
        top_experts = get_positive_experts(moe_model, moe_input_ids, moe_attention_mask)
        positive_masks = torch.zeros(current_batch_size, num_experts, dtype=torch.bool, device=device)
        for b in range(current_batch_size):
            positive_masks[b, top_experts[b]] = True
            
        # 提取文本喂给原装 Encoder (注意需要转成列表)
        raw_texts = batch["text"] 
        s_embs = f_s(raw_texts) 
        e_embs = f_e(expert_params)
        
        loss = bidirectional_contrastive_loss(s_embs, e_embs, positive_masks, tau=0.07)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"[Step {step}/{max_steps}] Loss: {loss.item():.4f}")
            
        step += 1
        
    # 保存至测试脚本默认读取的根目录
    torch.save({
        'sample_encoder_state': f_s.state_dict(),
        'expert_encoder_state': f_e.state_dict()
    }, output_path)
    print(f"\n=> 训练完成！对齐模型已保存至 {output_path}")

if __name__ == "__main__":
    train_ssa(
        moe_model_path="./Qwen1.5-MoE-A2.7B-Chat",
        minilm_path="./all-MiniLM-L6-v2",
        c4_path="/wenzhiquan/mr/datasets/c4/c4-train.00000-of-01024.json.gz", 
        output_path="./Sersa_outputs/final1.pt"
    )