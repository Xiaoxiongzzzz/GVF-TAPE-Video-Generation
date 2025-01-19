from unet import UnetLatent as Unet
from rectified_flow import RectifiedFlow
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader, Subset
from real_world_dataset import RealWorldDatasetLatent
from diffusers.models import AutoencoderKL
from peft import PeftConfig, PeftModel
from goal_diffusion import cycle
from einops import rearrange
from tqdm import tqdm
from torchvision.utils import save_image
import numpy as np
import torch
import wandb
import argparse
import os
import h5py
import imageio
import cv2

device = torch.device("cuda")
sample_timestep = 3
batch_text = ["open the drawer"]
def main():
    unet = Unet().to(device)
    unet.load_state_dict(torch.load("./results/RFlow_real_world/model_99000.pt")["model"])
    unet.eval()

    rectified_flow = RectifiedFlow(sample_timestep=sample_timestep)
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model).to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    vae.requires_grad_(False)
    
    batch_text_ids = tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(device)
    batch_text_embed = text_encoder(**batch_text_ids).last_hidden_state

    hdf5_file = h5py.File("/mnt/home/ZhangXiaoxiong/Data/0109_hardware_for_video_gen_test/0109_hardware_for_video_gen_test/episode_0.hdf5", 'r')
    initial_frame = torch.from_numpy(cv2.cvtColor(hdf5_file["observations"]["images"]["mid"][0], cv2.COLOR_BGR2RGB))/255.0
    initial_frame = initial_frame.unsqueeze(0).to(device).permute(0, 3, 1, 2)
    save_image(initial_frame, "./first_frame.png")
    hdf5_file.close()

    video_clip_list = []
    initial_frame_encoded = vae.encode(initial_frame).latent_dist.mean.mul_(vae.config.scaling_factor)
    
    for i in range(5):
        video_clip = sample_encoded_video(unet, rectified_flow, initial_frame_encoded, batch_text_embed)
        video_clip_list.append(video_clip)
        initial_frame_encoded = video_clip[:, -1] 

    encoded_video = torch.cat(video_clip_list, dim=1)
    video = decode_video(vae, encoded_video)

    with imageio.get_writer("./real_world.gif", mode='I', duration=100) as writer:
        for frame in video:
            writer.append_data(frame)
    
def sample_encoded_video(model, rectified_flow, x_cond, task_embed):
    with torch.no_grad():
        noise = torch.randn((1, 6*x_cond.shape[1], x_cond.shape[2], x_cond.shape[3]), device=device)
        sample = rectified_flow.sample(model, noise, x_cond.to(device), task_embed)
        sample = rearrange(sample, "b (f c) h w -> b f c h w", f=6)
    
    return sample 

def decode_video(vae, encoded_video):
    '''
    Args:
        encoded_video: (b, f, c, h, w)
    Returns:
        video: (f, h, w, c)
    '''
    encoded_video = encoded_video / vae.config.scaling_factor
    encoded_video = rearrange(encoded_video, "b f c h w -> (b f) c h w")
    video = vae.decode(encoded_video).sample
    video = (video).clamp(0, 1) * 255
    video = video.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)

    return video

if __name__ == "__main__":
    main()