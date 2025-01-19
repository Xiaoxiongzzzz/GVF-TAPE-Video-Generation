from goal_diffusion import GoalGaussianDiffusion, Trainer
from unet import UnetMW as Unet
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset, DataLoader, Subset
from peft import PeftConfig, PeftModel
from datasets import LiberoDatasetCloseLoop
from PIL import Image
from accelerate import Accelerator
import numpy as np
import torch.nn as nn
import torch
import os 
import json
import tqdm
import argparse
import wandb
def main(args):
    sample_per_seq = 7
    target_size = [128, 128]
    valid_n = 10
    interval = 4
    unet = Unet()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion = GoalGaussianDiffusion(
    channels=3*(sample_per_seq-1),
    model=unet,
    image_size=target_size,
    timesteps=100,
    sampling_timesteps=3,
    loss_type="l2",
    objective="pred_v",
    beta_schedule="cosine",
    min_snr_loss_weight=True,
    )

    
    train_set = LiberoDatasetCloseLoop(
    folder_path="/mnt/home/ZhangXiaoxiong/Data/atm_data/atm_libero/libero_spatial",
    sample_per_seq=sample_per_seq,
    target_size=target_size,
    interval=interval,
    view="side_view",
    train_ratio=1
    )
    valid_inds = [i for i in range(0, len(train_set), len(train_set)//valid_n)][:valid_n]
    valid_set = Subset(train_set, valid_inds)
    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        train_set=train_set,
        valid_set=valid_set,
        train_lr=1e-4,
        train_num_steps=100000,
        save_and_sample_every=3000,
        ema_update_every=10,
        ema_decay=0.999,
        train_batch_size=8,
        valid_batch_size=valid_n,
        gradient_accumulate_every=1,
        num_samples=valid_n, 
        results_folder='./results/ddpm_libero_spatial',
        fp16=True,
        amp=True,
        use_wandb=args.use_wandb,
    )
    if args.checkpoint_num is not None:
        trainer.load(args.checkpoint_num)
    
    if args.mode == 'train':
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'inference']) # 'train for training, 'inference' for generating samples
    parser.add_argument('-c', '--checkpoint_num', type=int, default=None) # checkpoint number to resume training or generate samples
    parser.add_argument('--use-wandb', type=bool, default=None)
    
    args = parser.parse_args()
    main(args)