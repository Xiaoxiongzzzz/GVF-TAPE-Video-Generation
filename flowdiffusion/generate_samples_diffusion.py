from torch.utils.data import Dataset, DataLoader, Subset
from datasets import LiberoDatasetCloseLoop
from goal_diffusion import GoalGaussianDiffusion
from transformers import CLIPTextModel, CLIPTokenizer
from unet import UnetLatent, UnetMW
from einops import rearrange
from tqdm import tqdm
import torch
import h5py

class Sample_Generator():
    def __init__(self):
        self.sample_per_seq = 7
        self.target_size = [128, 128]
        self.interval = 4
        self.depth = True
        self.batch_size = 16
        self.data_path = "/mnt/data0/xiaoxiong/atm_libero/libero_object"
        self.model_path = "/mnt/data0/xiaoxiong/single_view_goal_diffusion/diffusion_results/libero_object/DDPM_libero_object_scratch_90%_depth/ckpt/model_100000.pt"
        self.hdf5_file="/mnt/data0/xiaoxiong/FID/libero_object/DDPM_libero_object_scratch_timesteps_10.hdf5"
        
        self.device = torch.device("cuda")
        self.num_frames = 6  # Number of frames to generate
        self.sampling_timesteps = 10
        self.ratio = 0.1
        self.guidance_weight = 0.0  # 引导权重，默认为0

        self.setup_data()
        self.setup_models()
        
    def setup_data(self):
        self.data_set = LiberoDatasetCloseLoop(
            folder_path=self.data_path,
            sample_per_seq=self.sample_per_seq,
            target_size=self.target_size,
            interval=self.interval,
            depth=self.depth,
            train_ratio=self.ratio,
            mode="test"
        )
        self.data_loader = DataLoader(
            self.data_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )
        
    def setup_models(self):
        self.unet = UnetMW(depth=self.depth).to(self.device).requires_grad_(False)
        self.unet.eval()
        
        # 创建GoalGaussianDiffusion模型
        self.diffusion = GoalGaussianDiffusion(
            channels=4*(self.sample_per_seq-1),
            model=self.unet,
            image_size=self.target_size,
            timesteps=100,
            sampling_timesteps=self.sampling_timesteps,
            loss_type="l2",
            objective="pred_v",
            beta_schedule="cosine",
            min_snr_loss_weight=True,
        ).to(self.device).requires_grad_(False)
        self.diffusion.eval()  # 确保模型处于评估模式
        
        self.load_checkpoint()
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.model_path)
        model_checkpoint = checkpoint["model"]

        if any(k.startswith('module.') for k in model_checkpoint.keys()):
            model_checkpoint = {k.replace('module.', ''): v for k, v in model_checkpoint.items()}

        self.diffusion.load_state_dict(model_checkpoint) 

    def append_video_to_hdf5(self, video_dataset, videos):
        '''
        Append a batch of videos to the hdf5 dataset.
        
        Args:
            video_dataset: The hdf5 dataset to append to
            videos(np.array): A batch of videos to append (bs, t, c, h, w)
        '''
        current_size = video_dataset.shape[0]
        new_size = current_size + videos.shape[0]
        video_dataset.resize(new_size, axis=0)
        video_dataset[current_size:new_size] = videos

    def sample(self, x_cond, x_gt, task_embed):
        '''
        Generate a batch of sample.
        Args:
            x_cond(torch.Tensor): (Batch, C, H, W) (0-1)  
            x_gt(torch.Tensor): (Batch, T, C, H, W) (0-1)
            task_embedding(torch): (Batch, n, dim)
        Return:
            x_pred(torch.Tensor): (Batch, T, C, H, W)
        '''
        # 使用GoalGaussianDiffusion模型生成样本
        # 首先需要把x_gt变成正确的形状作为参考
        batch_size = x_cond.shape[0]
        
        # 使用模型的sample方法生成样本
        x_pred = self.diffusion.sample(
            x_cond=x_cond, 
            task_embed=task_embed, 
            batch_size=batch_size,
            guidance_weight=self.guidance_weight
        )
        
        # 重新排列维度，从(batch, frames*channels, h, w)到(batch, frames, channels, h, w)
        x_pred = rearrange(x_pred, "b (f c) h w -> b f c h w", f=self.num_frames)
        
        return x_pred

    def generate_and_save_samples(self):
        '''
        Loop the dataset, generate corresponding prediction and save them in a hdf5 file.
        Will stop when reaching maxshape limit.
        '''
        max_samples = 5000  # maxshape的第一维度
        current_samples = 0
        
        with h5py.File(self.hdf5_file, 'w') as f:
            prediction_dataset = f.create_dataset(
                'prediction',
                shape=(0, 6, 3, 128, 128),
                maxshape=(max_samples, 6, 3, 128, 128),
            )
            ground_truth_dataset = f.create_dataset(
                'ground_truth',
                shape=(0, 6, 3, 128, 128),
                maxshape=(max_samples, 6, 3, 128, 128),
            )
            
            for _, batch_data in enumerate(tqdm(self.data_loader)):
                # 检查是否达到最大样本数
                batch_size = batch_data[0].shape[0]
                if current_samples + batch_size > max_samples:
                    # 如果添加整个batch会超出限制，只取需要的部分
                    samples_needed = max_samples - current_samples
                    if samples_needed <= 0:
                        print(f"Reached maximum samples limit ({max_samples}). Stopping generation.")
                        break
                        
                    # 只取batch中的部分数据
                    x_gt, x_cond, task_embed = [data[:samples_needed] for data in batch_data]
                else:
                    x_gt, x_cond, task_embed = batch_data
                
                # 处理数据
                x_gt, x_cond, task_embed = \
                    map(lambda x: x.to(self.device), (x_gt, x_cond, task_embed))
                x_pred = self.sample(x_cond, x_gt, task_embed).cpu().detach().numpy()
                x_pred = x_pred[:,:,:3]  # 只保留RGB通道
                x_gt = x_gt.cpu().detach().numpy()
                x_gt = x_gt[:,:,:3]  # 只保留RGB通道

                # 添加到数据集
                self.append_video_to_hdf5(prediction_dataset, x_pred)
                self.append_video_to_hdf5(ground_truth_dataset, x_gt)
                
                # 更新计数
                current_samples += x_gt.shape[0]
                
                # 检查是否达到最大限制
                if current_samples >= max_samples:
                    print(f"Reached maximum samples limit ({max_samples}). Stopping generation.")
                    break


if __name__ == "__main__":
    sample_generator = Sample_Generator()
    sample_generator.generate_and_save_samples()