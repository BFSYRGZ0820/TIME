#!/bin/bash

# 设置下载存放的目录
DATA_DIR="/wenzhiquan/mr/datasets/c4"

# 如果目录不存在，则创建它
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}" || exit

echo "开始下载 C4 数据集的前 20 个分片..."

# 循环下载 00000 到 00019
for i in $(seq -f "%05g" 0 19); do
  echo "正在下载分片 ${i}..."
  curl -L "https://hf-mirror.com/datasets/allenai/c4/resolve/main/en/c4-train.${i}-of-01024.json.gz" -o "c4-train.${i}-of-01024.json.gz"
done

echo "下载完成！所有文件已保存在 ${DATA_DIR} 目录下。"