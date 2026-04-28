from modelscope import snapshot_download

# 定义需要下载的模型列表 (model_id: 魔搭上的模型名称, local_dir: 本地保存路径)
models_to_download = [
    {
        "model_id": "qwen/Qwen1.5-MoE-A2.7B",
        "local_dir": "./models/Qwen1.5-MoE-A2.7B"
    },
    {
        "model_id": "deepseek-ai/DeepSeek-V2-Lite",
        "local_dir": "./models/DeepSeek-V2-Lite"
    }
]

for model in models_to_download:
    print(f"========== 开始下载 {model['model_id']} ==========")
    model_dir = snapshot_download(
        model_id=model['model_id'],
        local_dir=model['local_dir'],
        revision='master' # 默认拉取最新版本
    )
    print(f"✅ 下载完成！模型已保存至: {model_dir}\n")

print("🎉 所有模型下载任务已全部完成！")