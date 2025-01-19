from torch.utils.data import Dataset
import os
from glob import glob
import torch
from utils import get_paths, get_paths_from_dir
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
import torchvision.transforms as T
import random
from torchvideotransforms import video_transforms, volume_transforms
from einops import rearrange
import h5py
import cv2

class RealWorldDataset(Dataset):
    def __init__ (self, folder_path, sample_per_seq=7, target_size=(128, 128), interval=4):
        self.folder_path = folder_path
        self.sample_per_seq = sample_per_seq
        self.interval = interval
        self.demo_list = []
        self.task_list = []

        self.hdf5_list = glob(os.path.join(folder_path, "**/*.hdf5"), recursive=True)
        for hdf5_file in self.hdf5_list:
            demos, tasks = self.get_demos_and_task_from_hdf5(hdf5_file)
            self.demo_list.extend(demos)
            self.task_list.extend(tasks)

        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor(),
            ])
        
        print("Total number of demos: ", len(self.demo_list))

    def get_demos_and_task_from_hdf5(self, hdf5_file):
        with h5py.File(hdf5_file, 'r') as f:
            demos = [f['observations']['images']['mid'][:]]
            tasks = [self.get_task_from_hdf5(hdf5_file)]

        return demos, tasks

    def get_task_from_hdf5(self, hdf5_file):
        task = hdf5_file.split("/")[-2].replace("_", " ")

        return task
    def get_seq_from_demo(self, demo):
        '''
        Args:
            demo: [T, H, W, C]
        '''
        horizon = demo.shape[0]

        start_index = np.random.randint(0, horizon-1-self.sample_per_seq-(self.sample_per_seq-1)*self.interval, 1)[0]
        seq = demo[start_index:start_index+self.sample_per_seq+(self.sample_per_seq-1)*self.interval:self.interval+1]

        return [cv2.cvtColor(s, cv2.COLOR_BGR2RGB) for s in seq]
    def __len__(self):
        return len(self.hdf5_list)

    def __getitem__(self, idx):
        demo = self.demo_list[idx]
        task = self.task_list[idx]
        
        video_clip = self.get_seq_from_demo(demo)
        video_clip = [self.transform(Image.fromarray(s)) for s in video_clip]
        video_clip = torch.stack(video_clip, dim=0)

        x_cond = video_clip[0]
        x = video_clip[1:]

        return x, x_cond, task

class RealWorldDatasetLatent(Dataset):
    def __init__ (self, folder_path, sample_per_seq=7, interval=4):
        self.folder_path = folder_path
        self.sample_per_seq = sample_per_seq
        self.interval = interval
        self.demo_list = []
        self.task_list = []

        self.hdf5_list = glob(os.path.join(folder_path, "**/*.hdf5"), recursive=True)
        for hdf5_file in self.hdf5_list:
            demos, tasks = self.get_demos_and_task_from_hdf5(hdf5_file)
            self.demo_list.extend(demos)
            self.task_list.extend(tasks)
        
        print("Total number of demos: ", len(self.demo_list))

    def get_demos_and_task_from_hdf5(self, hdf5_file):
        with h5py.File(hdf5_file, 'r') as f:
            demos = [f['latent_observation'][:]]
            tasks = [self.get_task_from_hdf5(hdf5_file)]

        return demos, tasks
    
    def get_task_from_hdf5(self, hdf5_file):
        task = hdf5_file.split("/")[-2].replace("_", " ")

        return task
    def get_seq_from_demo(self, demo):
        '''
        Args:
            demo: [T, H, W, C]
        '''
        horizon = demo.shape[0]

        start_index = np.random.randint(0, horizon-1-self.sample_per_seq-(self.sample_per_seq-1)*self.interval, 1)[0]
        seq = demo[start_index:start_index+self.sample_per_seq+(self.sample_per_seq-1)*self.interval:self.interval+1]

        return [torch.from_numpy(s) for s in seq]
    def __len__(self):
        return len(self.hdf5_list)

    def __getitem__(self, idx):
        demo = self.demo_list[idx]
        task = self.task_list[idx]
        
        video_clip = self.get_seq_from_demo(demo)
        video_clip = torch.stack(video_clip, dim=0)

        x_cond = video_clip[0]
        x = video_clip[1:]

        return x, x_cond, task

# # Test Code
# if __name__ == "__main__":
#     import torch
#     from diffusers.models import AutoencoderKL
#     from torchvision.utils import save_image
#     device = torch.device("cuda")
#     dataset = RealWorldDataset("/mnt/home/ZhangXiaoxiong/Data/real-data")
#     # vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
#     # vae.requires_grad_(False)
#     # vae.eval()

#     x, x_cond, task = dataset[0]
#     x_cond = x_cond.unsqueeze(0)
#     # x = vae.decode(x_cond).sample.cpu().detach()
#     save_image(x_cond, f"test.png")

#     import ipdb; ipdb.set_trace()
