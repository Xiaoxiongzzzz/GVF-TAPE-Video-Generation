from datasets import LiberoDatasetCloseLoop
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from peft import LoraConfig, get_peft_model
from torch.utils.data import Subset
from tqdm import tqdm
import numpy as np
import argparse
import torch    
import wandb
def main(args):
    total_epoch =50
    device = torch.device("cuda")
    if args.use_wandb:
        wandb.init(project="finetune-lora-vae",)

    vae = AutoencoderKL.from_pretrained('stabilityai/sdxl-vae')
    lora_config = LoraConfig( 
        r=16,
        lora_alpha=32,
        target_modules=["conv1", "conv2"],
        lora_dropout=0.05,
        bias="none",
    )
    lora_vae = get_peft_model(vae, lora_config)
    lora_vae = lora_vae.to(device)
    optimizer = torch.optim.AdamW(lora_vae.parameters(), lr=1e-5)

    train_set = LiberoDatasetCloseLoop(
                folder_path="/mnt/home/ZhangXiaoxiong/Data/atm_data/atm_libero",
                sample_per_seq=1,
                target_size=(128, 128),
                interval=0,
            )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    for epoch in range (total_epoch):
        with tqdm(total=len(train_loader), desc=f"{(epoch+1)}/{total_epoch}") as pbar:
            for batch in train_loader:
                eye_in_hand, side_view, _ = batch
                eye_in_hand, side_view = eye_in_hand.to(device), side_view.to(device)
                sample = torch.cat([eye_in_hand, side_view], dim=0)
                loss = compute_loss(lora_vae, sample)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.use_wandb:
                    wandb.log({"loss": loss.item()})
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
    lora_vae.save_pretrained("./lora-vae")
def compute_loss(model, sample):
    reconstruction = model(sample).sample
    loss = torch.nn.functional.mse_loss(reconstruction, sample, reduction="mean")

    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-wandb", type=bool, default=False)
    args = parser.parse_args()
    main(args)