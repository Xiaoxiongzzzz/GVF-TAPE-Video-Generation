from transformers import CLIPTokenizer, CLIPTextModel
import h5py
import torch
import os
import glob
FOLDER_PATH = "/home/ZhangChuye/Data/corl_tele_tasks"
MAX_LENGTH = 15
pretrained_model = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
text_encoder.requires_grad_(False)
text_encoder.eval()

# 获取所有任务文件夹
task_folders = [f for f in glob.glob(os.path.join(FOLDER_PATH, "*")) if os.path.isdir(f)]
all_embeddings = {}
for task_folder in task_folders:
    # 获取任务名称
    task_name = os.path.basename(task_folder)
    task_prompt = task_name.replace("_", " ")
    print(f"Processing task: {task_prompt}")
    
    # 为当前任务生成文本嵌入
    batch_text_ids = tokenizer(task_prompt, return_tensors='pt', padding=True, truncation=True, max_length=128)
    text_embed = text_encoder(**batch_text_ids).last_hidden_state

    
    padding_length = MAX_LENGTH - text_embed.shape[1]  
    if padding_length > 0:
        padding = torch.zeros((text_embed.shape[0], padding_length, text_embed.shape[2]), dtype=text_embed.dtype)
        text_embed = torch.cat((text_embed, padding), dim=1)
    
    all_embeddings[task_prompt] = text_embed.cpu().numpy()

with h5py.File("./real_world_text_embedding.hdf5", 'w') as f:
    # Create groups for texts and embeddings
    embedding_group = f.create_group('embeddings')
    
    # Store the data
    for i, (text, embedding) in enumerate(all_embeddings.items()):
        embedding_group.create_dataset(text, data=embedding)
    
    # Store metadata
    f.attrs['num_embeddings'] = len(all_embeddings)

print("Done!")


