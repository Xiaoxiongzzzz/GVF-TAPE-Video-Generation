from unet import UnetLatent, UnetMW
from rectified_flow import RectifiedFlow
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import LiberoDatasetCloseLoop
from diffusers.models import AutoencoderKL
from peft import PeftConfig, PeftModel
from goal_diffusion import cycle
from einops import rearrange
from tqdm import tqdm
from torchvision.utils import save_image
import torch
import wandb
import argparse
import os

device = torch.device("cuda")

class RectifiedFlowTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        self.sample_per_seq = 7
        self.save_every = 3000
        self.train_steps = 100000
        self.target_size = [128, 128]
        self.latent_size = [size // 8 for size in self.target_size]
        self.valid_n = 10
        self.interval = 4
        
        self.setup_wandb()
        self.setup_models()
        self.setup_data()
        
    def setup_wandb(self):
        if self.args.use_wandb:
            mode_name = 'latent' if self.args.latent_mode else 'pixel'
            wandb.init(project="AVDC", name=f"rectified-{mode_name}-flow-libero-goal")
            
    def setup_models(self):
        self.unet = (UnetLatent() if self.args.latent_mode else UnetMW()).to(self.device)
        self.load_checkpoint()
        
        self.rectified_flow = RectifiedFlow(sample_timestep=self.args.sample_step)
        self.setup_clip()
        self.setup_vae()
        
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.train_steps)
            
    def setup_clip(self):
        pretrained_model = "openai/clip-vit-base-patch32"
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model).to(self.device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        
    def load_checkpoint(self):
        self.initial_step = 0
        if self.args.model_path is not None:
            checkpoint = torch.load(self.args.model_path)
            self.unet.load_state_dict(checkpoint["model"])
            self.initial_step = checkpoint["step"]
            
    def setup_data(self):
        self.train_set = LiberoDatasetCloseLoop(
            folder_path="/mnt/home/ZhangXiaoxiong/Data/atm_data/atm_libero/v2a",
            sample_per_seq=self.sample_per_seq,
            target_size=self.target_size,
            interval=self.interval,
            latent=self.args.latent_mode,
            train_ratio=0.4,
        )
        self.train_loader = cycle(DataLoader(self.train_set, batch_size=8, shuffle=True, num_workers=8))

        valid_inds = [i for i in range(0, len(self.train_set), len(self.train_set)//self.valid_n)][:self.valid_n]
        self.valid_set = Subset(self.train_set, valid_inds)
        self.valid_set.augmentation = False
        self.valid_loader = cycle(DataLoader(self.valid_set, batch_size=self.valid_n, shuffle=False, num_workers=8))

    def setup_vae(self):
        if self.args.latent_mode:
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(self.device)
            self.vae.requires_grad_(False)
            self.vae.eval()

    def train_step(self, batch):
        x_start, x_cond, text = batch
        x_start, x_cond = x_start.to(self.device), x_cond.to(self.device)
        
        if self.args.latent_mode:
            x_start, x_cond = self.process_latent(x_start, x_cond)
        else:
            x_start = rearrange(x_start, "b f c h w -> b (f c) h w")
        
        task_embed = self.encode_batch_text(text)
        loss = self.rectified_flow.train_loss(self.unet, x_start, x_cond, task_embed)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        
        return loss.item()

    def process_latent(self, x, x_cond):
        """
        Input:
            x: [b, f, c, h, w]
            x_cond: [b, c, h, w]
        Return:
            x: [b, (fc), h, w]
            x_cond: [b, c, h, w]
        """
        if x is not None:
            x = x.mul_(self.vae.config.scaling_factor)
            x = rearrange(x, "b f c h w -> b (f c) h w")
        if x_cond is not None:
            x_cond = x_cond.mul_(self.vae.config.scaling_factor)

        return x, x_cond

    def train(self):
        for step in tqdm(range(self.initial_step + self.train_steps), initial=self.initial_step):
            loss = self.train_step(next(self.train_loader))
            
            if self.args.use_wandb:
                wandb.log({
                    "loss": loss,
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }, step=self.initial_step+step)
                
            if step % self.save_every == 0:
                self.sample_and_save(step)

    def encode_batch_text(self, batch_text):
        batch_text_ids = self.tokenizer(batch_text, return_tensors='pt', padding=True, 
                                      truncation=True, max_length=128).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
        return batch_text_embed

    @torch.no_grad()
    def sample_and_save(self, step, save_model=True):
        batch = next(self.valid_loader)
        x_start, x_cond, text = batch
        task_embed = self.encode_batch_text(text)
        x_start, x_cond = x_start.to(self.device), x_cond.to(self.device)
        B,F = x_start.shape[:2]

        if self.args.latent_mode:
            _, x_cond = self.process_latent(None, x_cond)
            noise_shape = (B, F*x_cond.shape[1], x_cond.shape[2], x_cond.shape[3])
        else:
            noise_shape = (B, F*x_cond.shape[1], x_cond.shape[2], x_cond.shape[3])

        noise = torch.randn(noise_shape, device=self.device)
        sample = self.rectified_flow.sample(self.unet, noise, x_cond, task_embed)
        
        if self.args.latent_mode:
            sample = rearrange(sample, "b (f c) h w -> (b f) c h w", c=4)
            sample = self.vae.decode(sample.mul_(1/self.vae.config.scaling_factor)).sample
            sample = rearrange(sample, "(b f) c h w -> b f c h w", b=B).cpu().detach()
            gt_start, gt_last = self.vae.decode(x_cond.mul_(1/self.vae.config.scaling_factor)).sample, self.vae.decode(x_start[:,-1]).sample
            gt_start, gt_last = gt_start.unsqueeze(1).cpu().detach(), gt_last.unsqueeze(1).cpu().detach()
        else:
            sample = rearrange(sample, "b (f c) h w -> b f c h w", f=6).cpu().detach()
            gt_start, gt_last = x_cond, x_start[:,-1]
            gt_start, gt_last = gt_start.unsqueeze(1).cpu().detach(), gt_last.unsqueeze(1).cpu().detach()

        images = torch.cat([gt_start, gt_last, sample], dim=1)
        n_row = images.shape[1]
        images = rearrange(images, "b f c h w -> (b f) c h w")

        os.makedirs(f"./results/RFlow_libero_90_{self.args.latent_mode}/", exist_ok=True)
        save_image(images, f"./results/RFlow_libero_90_{self.args.latent_mode}/sample_{step}.png", nrow=n_row)

        if save_model:
            data = {
                "step": step,
                "model": self.unet.state_dict(),
            }
            torch.save(data, f"./results/RFlow_libero_90_{self.args.latent_mode}/model_{step}.pt")

def main(args):
    trainer = RectifiedFlowTrainer(args)
    if args.mode == "train":
        trainer.train()
    elif args.mode == "inference":
        trainer.sample_and_save("test", save_model=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-wandb", default=None, type=bool)
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--model-path", default=None, type=str)
    parser.add_argument("--sample-step", default=3, type=int)
    parser.add_argument("--latent-mode", action="store_true")

    args = parser.parse_args()
    main(args)