import torch

# 提供给 utils/__init__.py 导出的占位张量
# 通常用于某些框架（如 DeepSpeed 或 Accelerate）在初始化计算图时的 dry-run 测试
DUMMY_INPUT_IDS = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
DUMMY_LABELS = torch.tensor([[-100, -100, 3, 4, 5]], dtype=torch.long)