from transformers import CLIPTokenizer, CLIPTextModel
from glob import glob
from tqdm import tqdm
import torch
import os
import h5py
MAX_LENGTH = 25
device = torch.device("cuda:4")
pretrained_model = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
text_encoder = CLIPTextModel.from_pretrained(pretrained_model).to(device)
text_encoder.requires_grad_(False)
text_encoder.eval()

folder_path = "/mnt/data0/xiaoxiong/atm_libero/libero_spatial_2"
file_path_list = glob(os.path.join(folder_path, "*.hdf5"), recursive=True)
for file_path in tqdm(file_path_list):
    text_prompt = os.path.basename(file_path).split(".")[0]
    text_prompt = text_prompt.replace("_", " ")[:-5]
    print(f"Processing {text_prompt}")
    with h5py.File(file_path, "r+") as f:
        data = f["data"] 
        text_inputs = tokenizer(text_prompt, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        text_embed = text_encoder(**text_inputs).last_hidden_state
        padding_length = MAX_LENGTH - text_embed.shape[1]  
        if padding_length > 0:
            padding = torch.zeros((text_embed.shape[0], padding_length, text_embed.shape[2]), dtype=text_embed.dtype, device=device)
            text_embed = torch.cat((text_embed, padding), dim=1)
       
        if "text_embed" in f.keys():
            del f["text_embed"]
        f["text_embed"] = text_embed.detach().cpu().numpy()
