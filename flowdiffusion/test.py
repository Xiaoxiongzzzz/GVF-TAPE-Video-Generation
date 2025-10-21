from datasets import LiberoDatasetCloseLoop
import torch  # 导入PyTorch

train_set = LiberoDatasetCloseLoop(
    folder_path="/home/ZhangChuye/Documents/vik_module/data/lb90_8tk_raw",
    sample_per_seq=7,
    target_size=[128, 128],
    interval=4,
    depth=True,
    train_ratio=0.4,
)
x, x_cond, task_embedding = train_set[0]

# 方法1：直接保存tensor
torch.save(task_embedding, 'task_embedding.pt')
