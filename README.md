# TiME 项目说明

本仓库是一个面向研究的实现，目标是在 Mixture-of-Experts (MoE) 大语言模型上进行持续测试时自适应（Continual Test-Time Adaptation, CTTA）。核心思想是在测试阶段在线调整路由行为与 LoRA 参数，并结合 Asymmetric COOT（As-COOT）进行样本-专家对齐。

## 1. 项目功能概览

- 加载本地 MoE 因果语言模型（默认 `Qwen1.5-MoE-A2.7B-Chat`）。
- 向 MoE 专家注入 LoRA 适配器。
- 冻结非 LoRA 参数，仅更新可训练部分。
- 在评测过程中执行持续测试时自适应。
- 使用 As-COOT 建立样本与专家之间的对齐关系。
- 通过前向 Hook 将对齐分布转化为路由偏置。
- 使用熵最小化与正则项联合优化测试时更新。

## 2. 核心流程

主入口脚本是 [moe-qwen.py](moe-qwen.py)：

1. 解析运行参数。
2. 从本地路径加载 tokenizer 和模型。
3. 对 MoE 模块注入 LoRA。
4. 为目标层注册路由 Hook。
5. 可选加载 Sersa 编码器检查点。
6. 运行持续测试时自适应流程。
7. 输出每个任务的汇总指标。

在 [module/AsCOOT-CTTA-MoE.py](module/AsCOOT-CTTA-MoE.py) 中，自适应主循环包括：

1. 使用 lm-eval API 对当前批次先做预评估。
2. 构建每个样本的任务上下文输入。
3. 编码样本表征与专家表征（若编码器可用）。
4. 求解 As-COOT 传输计划。
5. 基于传输质量过滤离群样本。
6. 将 OT 偏置注入 gate logits。
7. 执行熵最小化与参数正则更新。
8. 记录显存与耗时并汇总任务结果。

## 3. 目录结构说明

- [moe-qwen.py](moe-qwen.py)：主运行脚本。
- [module/As_COOT.py](module/As_COOT.py)：As-COOT 求解器。
- [module/AsCOOT-CTTA-MoE.py](module/AsCOOT-CTTA-MoE.py)：CTTA 主逻辑。
- [models/modeling_outputs.py](models/modeling_outputs.py)：模型输出数据结构。
- [utils/training_utils.py](utils/training_utils.py)：基于 accelerate 的训练工具。
- [static/finetune_config.yaml](static/finetune_config.yaml)：DeepSpeed 训练配置。
- [static/evaluation_config.yaml](static/evaluation_config.yaml)：DeepSpeed 评估配置。
- [scripts/run_moe.sh](scripts/run_moe.sh)：集群脚本示例。

## 4. 环境管理（Conda）

推荐使用 Python 3.10。


### 手动创建 + requirements

```bash
conda create -n time-moe python=3.10 -y
conda activate time-moe
pip install --upgrade pip
pip install -r requirements.txt
```

说明：

- [requirements.txt](requirements.txt) 中已包含主要依赖。
- Windows 下通常不建议直接安装 DeepSpeed，建议在 Linux/WSL 使用。

## 5. 运行前准备

1. 本地模型目录（默认）：
   - `./Qwen1.5-MoE-A2.7B-Chat`
2. 可选 Sersa 编码器检查点：
   - `./Sersa_outputs/sersa_5000_ckpt/final.pt`
3. 所选 lm-eval 任务所需的数据资源。

如果缺少可选编码器检查点，程序会打印 warning 并跳过对应逻辑。

## 6. 运行方式

### A. 直接运行（最简单）

```bash
python moe-qwen.py \
  --model_name="./Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande" \
  --train_batch_size=4 \
  --eval_batch_size=4 \
  --epsilon=0.01 \
  --ot_strength=5.0 \
  --lr=1e-5 \
  --result_path="results/results_qwen_test.txt" \
  --output_path="results/qwen/test"
```

### B. Accelerate + DeepSpeed（推荐 Linux/WSL）

```bash
accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 0 moe-qwen.py \
  --model_name="./Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq" \
  --train_batch_size=8 \
  --eval_batch_size=8 \
  --epsilon=0.01 \
  --ot_strength=5.0 \
  --lr=1e-5 \
  --result_path="results/results_qwen_test.txt" \
  --output_path="results/qwen/test"
```

### C. 集群脚本（Slurm）

```bash
bash scripts/run_moe.sh
```

## 7. 输出结果

- 控制台输出每个任务的汇总结果：准确率、显存占用、单位样本耗时。
- 可选文本结果文件（`result_path`）。
- 可选模型输出目录（`output_path`）。

## 8. 已知问题与注意事项

1. Python 导入命名问题：
   - `from module.AsCOOT-CTTA-MoE import ...` 不是合法 Python 模块导入（模块名不能含 `-`）。
2. 当前快照中存在被引用但未出现的模块文件：
   - `evaluation.py`
   - `module/moe_lora.py`
   - `module/Sersa_module.py`
   - `module/Expert_encoder_cache.py`
   - `module/Construct_context.py`
   - `utils/dummy_test.py`

若这些文件在你的本地仓库中缺失，运行时会发生导入错误。

## 9. 复现实验建议

- 固定模型版本和 tokenizer 版本。
- 保持随机种子一致。
- 记录 torch、transformers、lm-eval、CUDA 版本。
- 保持任务列表、批大小和超参数一致。
