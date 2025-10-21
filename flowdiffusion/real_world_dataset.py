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
from einops import rearrange
import h5py
import cv2

class RealWorldDataset(Dataset):
    def __init__(self, folder_path,
                 sample_per_seq=7,
                 target_size=(128, 128),
                 interval=4,
                 depth=True,
                 train_ratio=0.4,
                 mode="train"):
        print("Preparing dataset...")
        self.folder_path = folder_path
        self.sample_per_seq = sample_per_seq
        self.interval = interval
        self.depth = depth
        self.train_ratio = train_ratio
        self.mode = mode
        self.demo_list = []
        self.task_list = []
        self.valid_indices = []  # (demo_idx, start_idx)

        # 获取所有任务文件夹
        task_folders = [f for f in glob(os.path.join(folder_path, "*")) if os.path.isdir(f)]
        
        for task_folder in task_folders:
            # 获取当前任务文件夹下的所有hdf5文件
            task_hdf5_files = glob(os.path.join(task_folder, "*.hdf5"))
            num_files = int(len(task_hdf5_files) * self.train_ratio)
            
            # 根据mode选择对应的文件
            if self.mode == "train":
                selected_files = task_hdf5_files[:num_files]
            else:
                selected_files = task_hdf5_files[-num_files:]
            
            # 处理选中的文件
            for hdf5_file in selected_files:
                demo, tasks = self.get_demos_and_task_from_hdf5(hdf5_file)
                
                # Calculate valid starting indices for each demo
                demo_offset = len(self.demo_list)
                self._add_valid_indices(demo, demo_idx=demo_offset)
                
                self.demo_list.append(demo)
                self.task_list.append(tasks)

        self.resize = T.Resize(target_size)
        
        print(f"Total number of demos: {len(self.demo_list)}")
        print(f"Total number of valid sequences: {len(self.valid_indices)}")
        print("Done")

    def _add_valid_indices(self, demo, demo_idx):
        """Add valid starting indices for a demo to the valid_indices list."""
        horizon = demo.shape[0]
        seq_length = self.sample_per_seq + (self.sample_per_seq - 1) * self.interval
        
        for start_idx in range(horizon - seq_length + 1):
            self.valid_indices.append((demo_idx, start_idx))

    def get_demos_and_task_from_hdf5(self, hdf5_file):
        with h5py.File(hdf5_file, 'r') as f:
            demo = f['observations']['images']['mid'][:]
            demo_depth = f['observations']['depth']['mid'][:]
            demo = np.concatenate([demo, demo_depth[:,:,:,None]], axis=3)

            tasks = f['text_embed'][0]
        return demo, tasks

    def get_seq_from_demo(self, demo, start_index):
        """
        Args:
            demo: [T, H, W, C]
            start_index: The starting index for sampling
        Return:
            seq(np.array): [T, H, W, C]
        """
        end_index = start_index + self.sample_per_seq + (self.sample_per_seq - 1) * self.interval
        seq = demo[start_index:end_index:self.interval+1]
        return seq

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Returns:
            x: [T-1, C, H, W]
            x_cond: [C, H, W]
            task: [n, c]
        """
        demo_idx, start_idx = self.valid_indices[idx]
        
        demo = self.demo_list[demo_idx]
        task = self.task_list[demo_idx]
        
        video_clip = self.get_seq_from_demo(demo, start_idx)
        video_clip = (torch.from_numpy(video_clip)).permute(0, 3, 1, 2)/255.0
        video_clip = self.resize(video_clip)
        #Convert BGRD to RGBD
        video_clip = torch.cat([video_clip[:, [2, 1, 0], :, :],
                                 video_clip[:, 3:, :, :]], dim=1)
        
        x_cond = video_clip[0]
        x_cond = x_cond[:3]

        x = video_clip[1:]
        if not self.depth:
            x = x[:, :3, :, :]

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
#     from torchvision.utils import save_image
#     train_dataset = RealWorldDataset(
#         folder_path="/home/ZhangChuye/Data/CORL_HAND",
#         sample_per_seq=7,
#         target_size=(224, 224),
#         interval=1,
#         train_ratio=0.1,
#         mode="train"
#     )
#     x, x_cond, task = train_dataset[0]
#     print(x.shape)
#     print(x_cond.shape)
#     print(task.shape)
#     save_image(x[:, :3, :, :], f"test.png")
