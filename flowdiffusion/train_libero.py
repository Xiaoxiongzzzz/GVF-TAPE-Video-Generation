from goal_diffusion import GoalGaussianDiffusion, Trainer
from unet import UnetLatent, UnetMW
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset, DataLoader, Subset
from peft import PeftConfig, PeftModel
from datasets import LiberoDatasetCloseLoop
from PIL import Image
from accelerate import Accelerator
import numpy as np
import torch.nn as nn
import argparse
import wandb
def main(args):
    sample_per_seq = 7
    target_size = [128, 128]
    valid_n = 10
    interval = 4
    depth = True
    save_and_sample_every = 5000
    train_num_steps = 100000
    train_batch_size = 8
    unet = UnetMW(depth=depth)
    train_ratio=0.4

    # pretrained_model = "openai/clip-vit-base-patch32"
    # tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    # text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    # text_encoder.requires_grad_(False)
    # text_encoder.eval()

    diffusion = GoalGaussianDiffusion(
    channels=4*(sample_per_seq-1),
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
    folder_path="/home/ZhangChuye/Documents/vik_module/data/lb90_8tk_raw",
    sample_per_seq=sample_per_seq,
    target_size=target_size,
    interval=interval,
    depth=depth,
    train_ratio=train_ratio,
    )

    valid_inds = [i for i in range(0, len(train_set), len(train_set)//valid_n)][:valid_n]
    valid_set = Subset(train_set, valid_inds)
    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=None,
        text_encoder=None,
        train_set=train_set,
        valid_set=valid_set,
        train_lr=1e-4,
        train_num_steps=train_num_steps,
        save_and_sample_every=save_and_sample_every,
        train_batch_size=train_batch_size,
        valid_batch_size=valid_n,
        num_samples=valid_n, 
        results_folder='/mnt/data0/xiaoxiong/single_view_goal_diffusion/diffusion_results/v2a/DDPM_lb_8tasks_100%_100k',
        fp16=False,
        amp=False,
        use_wandb=args.use_wandb,
        depth=depth
    )
    if args.checkpoint_num is not None:
        trainer.load(args.checkpoint_num)
    print(trainer.model)
    if args.mode == 'train':
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'inference']) # 'train for training, 'inference' for generating samples
    parser.add_argument('-c', '--checkpoint_num', type=int, default=None) # checkpoint number to resume training or generate samples
    parser.add_argument('--use-wandb', action='store_true', default=False)
    
    args = parser.parse_args()
    main(args)