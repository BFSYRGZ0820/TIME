import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
from datetime import datetime
import torch
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.evaluator import simple_evaluate
from lm_eval.utils import make_table

def run_baseline(
    model_name="./Qwen1.5-MoE-A2.7B-Chat",
    task="winogrande,arc_challenge,arc_easy,boolq,openbookqa,rte,xquad_zh,xquad_es",
    eval_batch_size=8,
    result_path="results/baseline_results.txt",
    max_examples_per_task=0,
):
    """
    运行 Zero-Shot baseline 评测。

    命令行参数（通过 Fire 自动暴露）：
      --model_name   模型路径
      --task         逗号分隔任务列表
      --eval_batch_size 评测 batch size
      --result_path  结果 txt 输出路径
      --max_examples_per_task 每个 task 使用的样本数，<=0 表示全量
    """
    start_dt = datetime.now()
    start_ts = time.time()

    if isinstance(task, (tuple, list)):
        task_list = [str(t).strip() for t in task]
    else:
        task_list = [t.strip() for t in str(task).split(",") if t.strip()]

    print("========== 🚀 Starting Zero-Shot Baseline Evaluation ==========")
    print(f"Start Time:   {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Path:   {model_name}")
    print(f"Task:         {','.join(task_list)}")
    print(f"Batch Size:   {eval_batch_size}")
    print(f"Max Examples: {max_examples_per_task}")
    print(f"Result Path:  {result_path}")
    print("=============================================================")
    
    # 1. 加载最原始的底座模型（没有任何 LoRA 和 As-COOT 挂件）
    print("\n[1/3] Loading Model & Tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", # 使用 auto，但推荐启动时限制单卡 CUDA_VISIBLE_DEVICES=0
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 封装为 lm_eval 官方支持的模型对象
    print("\n[2/3] Wrapping model for lm_eval...")
    lm_obj = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=eval_batch_size
    )
    
    # 3. 启动标准评测引擎 (强制 Zero-shot)
    print(f"\n[3/3] Running evaluation for tasks: {','.join(task_list)}...")
    task_manager = lm_eval.tasks.TaskManager()
    if max_examples_per_task is None or int(max_examples_per_task) <= 0:
        results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=task_list,
            num_fewshot=0,
            task_manager=task_manager,
            batch_size=eval_batch_size
        )
    else:
        task_results = {}
        configs = {}
        versions = {}
        n_shot = {}
        higher_is_better = {}

        for task_name in task_list:
            print(f"[Baseline] Evaluating task {task_name} with max_examples_per_task={max_examples_per_task}")
            per_task = lm_eval.simple_evaluate(
                model=lm_obj,
                tasks=[task_name],
                num_fewshot=0,
                task_manager=task_manager,
                batch_size=eval_batch_size,
                limit=int(max_examples_per_task),
            )
            task_results.update(per_task.get("results", {}))
            configs.update(per_task.get("configs", {}))
            versions.update(per_task.get("versions", {}))
            n_shot.update(per_task.get("n-shot", {}))
            higher_is_better.update(per_task.get("higher_is_better", {}))

        results = {
            "results": task_results,
            "configs": configs,
            "versions": versions,
            "n-shot": n_shot,
            "higher_is_better": higher_is_better,
        }
    
    # 4. 打印并保存最终成绩单
    print("\n" + "="*20 + "  Baseline Results " + "="*20)
    metric_str = lm_eval.utils.make_table(results)
    print(metric_str)

    end_dt = datetime.now()
    end_ts = time.time()
    total_seconds = end_ts - start_ts

    if torch.cuda.is_available():
        peak_allocated_bytes = torch.cuda.max_memory_allocated()
        peak_reserved_bytes = torch.cuda.max_memory_reserved()
    else:
        peak_allocated_bytes = 0
        peak_reserved_bytes = 0
    peak_allocated_gb = peak_allocated_bytes / (1024 ** 3)
    peak_reserved_gb = peak_reserved_bytes / (1024 ** 3)
    
    # 确保存档目录存在
    result_dir = os.path.dirname(result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    
    # 仅将摘要信息和最终评测表格写入 txt
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("Baseline Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Start Time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Time (s): {total_seconds:.2f}\n")
        f.write(f"Peak GPU Memory Allocated (GB): {peak_allocated_gb:.2f}\n")
        f.write(f"Peak GPU Memory Reserved (GB):  {peak_reserved_gb:.2f}\n")
        f.write(f"Model Path: {model_name}\n")
        f.write(f"Tasks: {','.join(task_list)}\n")
        f.write(f"Eval Batch Size: {eval_batch_size}\n")
        f.write(f"Max Examples Per Task: {max_examples_per_task}\n")
        f.write(f"Result Path: {result_path}\n")
        f.write("="*50 + "\n")
        f.write(metric_str)
        f.write("\n")
        
    print("\n" + "="*20 + "  Run Summary " + "="*20)
    print(f"End Time:     {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Time:   {total_seconds:.2f}s")
    print(f"Peak Alloc:   {peak_allocated_gb:.2f} GB")
    print(f"Peak Reserved:{peak_reserved_gb:.2f} GB")
    print(f"\n All done! Results saved to: {result_path}")

if __name__ == "__main__":
    Fire(run_baseline)
