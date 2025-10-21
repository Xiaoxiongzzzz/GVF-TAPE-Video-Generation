from unet import UnetLatent, UnetMW
from rectified_flow import RectifiedFlow
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import LiberoDatasetCloseLoop
from diffusers.models import AutoencoderKL
from goal_diffusion import cycle
from einops import rearrange
from tqdm import tqdm
from torchvision.utils import save_image
import torch
import wandb
import argparse
import os
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm import tqdm

device = torch.device("cuda")

class RectifiedFlowTrainer:
    def __init__(self, args):
        self.args = args
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=1,
            log_with="wandb" if args.use_wandb else None,
            split_batches=True,
            step_scheduler_with_optimizer=False,
        )
        self.device = self.accelerator.device
        
        # Set random seed
        if args.seed is not None:
            set_seed(args.seed)
        
        # Setup logging
        self.logger = get_logger(__name__)
        self.num_process = self.accelerator.num_processes

        # Model parameters
        self.sample_per_seq = 7
        self.save_every = 5000
        self.train_steps = 100000
        self.target_size = [128, 128]
        self.valid_n = 8
        self.interval = 4
        self.depth = args.depth
        self.learning_rate = 1e-4
        self.batch_size = 8
        self.data_path = "/mnt/data0/xiaoxiong/atm_libero/libero_spatial"

        # Video generation parameters
        self.num_frames = 6  # Number of frames to generate
        self.train_ratio = 0.9

        # Save paths    
        self.results_base_dir = "/mnt/data0/xiaoxiong/single_view_goal_diffusion/results/libero_spatial"
        self.model_name = "RFlow_libero_spatial_100k_90%"
        
        
        self.setup_wandb()
        self.setup_models()
        self.setup_data()
        
    def setup_wandb(self):
        if self.args.use_wandb:
            full_config = {
                "args": vars(self.args),
                
                "model": {
                    "depth": self.depth,
                    "num_frames": self.num_frames,
                    "learning_rate": self.learning_rate,
                    "train_steps": self.train_steps,
                    "batch_size": self.batch_size,
                    "num_gpus": self.num_process,
                    "effective_batch": self.batch_size * self.num_process,
                },
                
                "data": {
                    "target_size": self.target_size,
                    "sample_per_seq": self.sample_per_seq,
                    "interval": self.interval,
                    "train_ratio": self.train_ratio,
                    "data_path": self.data_path,
                },
                
                "training": {
                    "save_every": self.save_every,
                    "scheduler_T_max": self.train_steps * self.num_process,
                    "seed": self.args.seed,
                }
            }
            
            self.accelerator.init_trackers(
                project_name="AVDC",
                config=full_config,
                init_kwargs={"wandb": {
                    "name": f"rectified-flow-{'rgbd' if self.depth else 'rgb'}-{self.num_process}gpus",
                    "tags": ["multi-gpu", "rectified-flow", "libero"],
                }}
            )
            
    def setup_models(self):
        # 1. First initialize model
        self.unet = UnetMW(depth=self.depth)
        
        # 2. Initialize optimizer and scheduler (but don't prepare them yet)
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.train_steps 
        )
        
        # 3. Load checkpoint (this will update model, optimizer and scheduler states)
        self.load_checkpoint()
        
        # 4. Setup other models
        self.rectified_flow = RectifiedFlow(sample_timestep=self.args.sample_step)
        self.setup_clip()
        
        # 5. Finally prepare all models for distributed training
        (
            self.unet,
            self.optimizer,
            self.lr_scheduler,
            self.text_encoder
        ) = self.accelerator.prepare(
            self.unet,
            self.optimizer,
            self.lr_scheduler,
            self.text_encoder
        )
            
    def setup_clip(self):
        pretrained_model = "openai/clip-vit-base-patch32"
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model).to(self.device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        
    def load_checkpoint(self):
        self.initial_step = 0
        if self.args.model_path is not None:
            print(f"Loading checkpoint from {self.args.model_path}")
            checkpoint = torch.load(self.args.model_path)
            # Load model state
            model_checkpoint = checkpoint["model"]

            if any(k.startswith('module.') for k in model_checkpoint.keys()):
                model_checkpoint = {k.replace('module.', ''): v for k, v in model_checkpoint.items()}

            self.accelerator.wait_for_everyone()

            self.unet.load_state_dict(model_checkpoint)
            
            if not self.args.fine_tune:  # Check if fine-tuning is enabled
                self.initial_step = checkpoint["step"]
                print(f"Restored model state from step {self.initial_step}")
                
                # Load optimizer state
                if "optimizer" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                    print("Restored optimizer state")
                
                # Load scheduler state
                if "scheduler" in checkpoint:
                    self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
                    print("Restored scheduler state")
            
    def setup_data(self):
        self.train_set = LiberoDatasetCloseLoop(
            folder_path=self.data_path,
            sample_per_seq=self.sample_per_seq,
            target_size=self.target_size,
            interval=self.interval,
            depth=self.depth,
            train_ratio=self.train_ratio,
        )
        train_dataloader = DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=8
        )
        
        # Prepare dataloader for distributed training
        self.train_loader = self.accelerator.prepare(train_dataloader)
        self.train_loader = cycle(self.train_loader)

        # Prepare validation dataloader
        valid_inds = [i for i in range(0, len(self.train_set), len(self.train_set)//self.valid_n)][:self.valid_n]
        self.valid_set = Subset(self.train_set, valid_inds)
        self.valid_set.augmentation = False
        valid_dataloader = DataLoader(
            self.valid_set, 
            batch_size=self.valid_n, 
            shuffle=False, 
            num_workers=8
        )
        self.valid_loader = self.accelerator.prepare(valid_dataloader)
        self.valid_loader = cycle(self.valid_loader)

    def train_step(self, batch):
        x_start, x_cond, task_embed = batch
        x_start = rearrange(x_start, "b f c h w -> b (f c) h w")
        
        with self.accelerator.accumulate(self.unet):
            loss = self.rectified_flow.train_loss(self.unet, x_start, x_cond, task_embed)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item()

    def train(self):        
        disable_progress_bar = not self.accelerator.is_local_main_process
        progress_bar = tqdm(
            range(self.initial_step, self.train_steps),
            disable=disable_progress_bar
        )
        
        for step in progress_bar:
            loss = self.train_step(next(self.train_loader))
            
            if self.accelerator.is_main_process:
                progress_bar.set_postfix(loss=f"{loss:.4f}")
                
                if self.args.use_wandb:
                    self.accelerator.log({
                        "loss": loss,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    }, step=step)
                    
                if step % self.save_every == 0:
                    self.sample_and_save(step)
        
        # Save the last model
        if self.accelerator.is_main_process:
            self.sample_and_save(self.train_steps)  

    def encode_batch_text(self, batch_text):
        batch_text_ids = self.tokenizer(batch_text, return_tensors='pt', padding=True, 
                                      truncation=True, max_length=128).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
        return batch_text_embed

    @torch.no_grad()
    def sample_and_save(self, step, save_model=True):
        # Only save on main process
        if not self.accelerator.is_main_process:
            return
            
        batch = next(self.valid_loader)
        x_start, x_cond, task_embed = batch
        x_start, x_cond, task_embed = \
             x_start.to(self.device), x_cond.to(self.device), task_embed.to(self.device)
        B,F = x_start.shape[:2]

        noise = torch.randn_like(x_start, device=self.device)
        noise = rearrange(noise, "b f c h w -> b (f c) h w")
        sample = self.rectified_flow.sample(self.unet, noise, x_cond, task_embed)
        
        # Reshape sample using class parameters
        sample = rearrange(sample, "b (f c) h w -> b f c h w", f=self.num_frames).cpu().detach()
        
        # Process ground truth
        gt_start, gt_last = x_cond, x_start[:,-1]
        gt_start = gt_start.unsqueeze(1).cpu().detach()  # RGB input only
        gt_last = gt_last.unsqueeze(1).cpu().detach()
        
        if self.depth:
            # Split RGB and depth channels from generated samples
            rgb_sample = sample[:, :, :3]
            depth_sample = sample[:, :, 3:4]
            
            # Process RGB channels
            rgb_gt_start = gt_start  # Use directly as it's RGB only
            rgb_gt_last = gt_last[:, :, :3]  # Get RGB from gt_last
            
            # Get ground truth depth from x_start
            depth_gt_last = gt_last[:, :, 3:4]  # Extract depth channel from gt_last
            
            # Concatenate RGB images (condition, target RGB, generated RGB)
            rgb_images = torch.cat([rgb_gt_start, rgb_gt_last, rgb_sample], dim=1)
            # Concatenate depth images (ground truth depth, generated depth)
            depth_images = torch.cat([depth_gt_last, depth_sample], dim=1)
            
            # Prepare for visualization
            n_row = rgb_images.shape[1]
            n_row_depth = depth_images.shape[1]
            rgb_images = rearrange(rgb_images, "b f c h w -> (b f) c h w")
            depth_images = rearrange(depth_images, "b f c h w -> (b f) c h w")
        else:
            # For RGB only mode
            images = torch.cat([gt_start, gt_last, sample], dim=1)
            n_row = images.shape[1]
            images = rearrange(images, "b f c h w -> (b f) c h w")

        # Create output directories
        save_dir = os.path.join(self.results_base_dir, 
                              f"{self.model_name}{'_depth' if self.depth else ''}")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "ckpt"), exist_ok=True)

        # Save visualization results
        if self.depth:
            save_image(rgb_images, os.path.join(save_dir, f"sample_rgb_{step}.png"), nrow=n_row)
            save_image(depth_images, os.path.join(save_dir, f"sample_depth_{step}.png"), nrow=n_row_depth)
        else:
            save_image(images, os.path.join(save_dir, f"sample_{step}.png"), nrow=n_row)

        # Save model checkpoint
        if save_model:
            data = {
                "step": step,
                "model": self.accelerator.get_state_dict(self.unet),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.lr_scheduler.state_dict(),
            }
            torch.save(data, os.path.join(save_dir, "ckpt", f"model_{step}.pt"))

def main(args):
    trainer = RectifiedFlowTrainer(args)
    if args.mode == "train":
        trainer.train()
    elif args.mode == "inference":
        trainer.sample_and_save("test", save_model=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--model-path", default=None, type=str)
    parser.add_argument("--sample-step", default=3, type=int)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fine-tune", action="store_true")
    
    args = parser.parse_args()
    main(args)