def evaluate_fewshot(*args, **kwargs):
    """
    占位函数。
    在当前的 TiME (CTTA) 主流程中，实际的 few-shot 评测
    已经交由 AsCOOT-CTTA-MoE.py 中的 lm_eval.evaluator.simple_evaluate 处理。
    此函数仅为防止 moe-qwen.py 发生 ImportError 而保留。
    """
    pass

def get_calib_dataloder(*args, **kwargs):
    """
    占位函数。
    通常用于 PTQ 量化或模型剪枝前获取校准（Calibration）数据集。
    在线测试时自适应（CTTA）流程不需要此功能，
    仅为防止 moe-qwen.py 发生 ImportError 而保留。
    """
    return []