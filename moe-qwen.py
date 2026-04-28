
# -*- coding: utf-8 -*-
import json
import os
import time
from typing import Optional
import logging
import torch
from fire import Fire
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluation import evaluate_fewshot, get_calib_dataloder
from module.moe_lora import inject_lora_into_moe, freeze_non_lora_params, register_layer_router_hook
from torch.utils.data import Dataset, DataLoader
from module.AsCOOT_CTTA_MoE import continual_test_time_adaptation
from module.Sersa_module import ExpertEncoder, TrainableSampleEncoder, extract_adapter_params_for_experts
logger = logging.getLogger(__name__)

class Args:
    def __init__(
        self,
        task,
        model_name: Optional[str] = "./Qwen1.5-MoE-A2.7B-Chat",
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 4,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        num_fewshot: Optional[int] = 0,
        epsilon: Optional[float] = 0.01,
        PI_OUTLIER_RATIO: Optional[float] = 0.01,
        reg_lambda: Optional[float] = 0.1,
        ot_strength: Optional[float] = 1.0,
        lr: Optional[float] = 1e-5,
    ):
        self.task = task
        self.model_name = model_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.output_path = output_path
        self.result_path = result_path
        self.model_path = model_path
        self.num_fewshot = num_fewshot
        self.epsilon = epsilon
        self.PI_OUTLIER_RATIO = PI_OUTLIER_RATIO
        self.reg_lambda = reg_lambda
        self.ot_strength = ot_strength
        self.lr=lr


class C4Dataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512, max_samples=None):
        self.texts = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                obj = json.loads(line)
                self.texts.append(obj.get('text', '')[:20000])
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(text, truncation=True,padding="max_length", max_length=self.max_len, return_tensors='pt')
        return {'input_ids': enc['input_ids'][0], 'attention_mask': enc['attention_mask'][0]}

def run_moe(
        task: str,
        model_name: Optional[str] = "./Qwen1.5-MoE-A2.7B-Chat",
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 4,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        num_fewshot: Optional[int] = 0,
        epsilon: Optional[float] = 0.01,
        PI_OUTLIER_RATIO: Optional[float] = 0.01,
        reg_lambda: Optional[float] = 0.1,
        ot_strength: Optional[float] = 1.0,
        lr: Optional[float] = 1e-5,
):
    ### Initialization
    torch.manual_seed(0)

    args = Args(
        task=task,
        model_name=model_name,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        output_path=output_path,
        result_path=result_path,
        model_path=model_path,
        num_fewshot=num_fewshot,
        epsilon = epsilon,
        PI_OUTLIER_RATIO = PI_OUTLIER_RATIO,
        reg_lambda = reg_lambda,
        ot_strength = ot_strength,
        lr=lr
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    ### Injecting LoRA
    model = inject_lora_into_moe(model, r=16, alpha=16)

    ### Freezing grad
    freeze_non_lora_params(model)

    ### Hooking routed experts
    routed_experts = {}
    register_layer_router_hook(model, routed_experts, layers_to_track=[0], topk=2)


    # ### CTTA-MoE
    ckpt_path = f"./Sersa_outputs/final.pt"
    enc_ckpt = None
    if os.path.exists(ckpt_path):
        enc_ckpt = torch.load(ckpt_path, map_location="cpu")
    sample_encoder = None
    expert_encoder = None
    if enc_ckpt is not None:
        adapter_map = extract_adapter_params_for_experts(model, layers_to_use=[0])
        _, first_vec = next(iter(adapter_map.items()))
        #expert_input_dim = first_vec.shape[0]
        expert_input_dim = first_vec.shape[1]
        expert_encoder = ExpertEncoder(input_dim=expert_input_dim, embedding_dim=256, device="cpu")
        sample_encoder = TrainableSampleEncoder(encoder_name='./all-MiniLM-L6-v2', embedding_dim=256, device="cpu")
        if 'expert_encoder_state' in enc_ckpt:
            expert_encoder.load_state_dict(enc_ckpt['expert_encoder_state'])
        else:
            print("not found expert_encoder_state")
        if 'sample_encoder_state' in enc_ckpt:
            sample_encoder.load_state_dict(enc_ckpt['sample_encoder_state'])
        else:
            print("not found sample_encoder_state")
        expert_encoder.eval()
        sample_encoder.eval()
    else:
        print("[Warning] No Sersa encoder checkpoint found; COOT_CTTA disabled unless you provide encoders.")

    start_time = time.time()
    all_task_results = continual_test_time_adaptation(args, model,
                                                      tokenizer, routed_experts,
                                                      sample_encoder=sample_encoder,
                                                      expert_encoder=expert_encoder,
                                                      layers_to_use=[0])
    print(f"[TTA] Results: {all_task_results} | Time: {time.time() - start_time:.2f} seconds | Number of parameters: {model.num_parameters()}")


    ### Save model
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # torch.save(model.state_dict(), output_path+"/model.pth")

if __name__ == "__main__":
    Fire(run_moe)




