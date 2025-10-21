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

### Sequential Datasets: given first frame, predict all the future frames
class LiberoDatasetCloseLoop(Dataset):
    def __init__(self, folder_path="/mnt/home/ZhangXiaoxiong/Data/atm_data/atm_libero/libero_spatial",
                 sample_per_seq=7,
                 target_size=(128, 128),
                 interval=1,
                 depth=True,
                 train_ratio=0.4,
                 mode="train"):
        print("Preparing dataset...")
        self.sideview_demo_list = []
        self.task_embed_list = []
        self.sample_per_seq = sample_per_seq
        self.interval = interval
        self.depth = depth 
        self.train_ratio = train_ratio
        self.mode = mode
        
        # Store indices for valid sampling points
        self.valid_indices = []  # (demo_idx, start_idx)
        
        hdf5_files = glob(os.path.join(folder_path, "./*.hdf5"), recursive=True)
        for hdf5_file in hdf5_files:
            sideview_demos, tasks_embeds = self.get_demos_and_task_from_hdf5(hdf5_file)
            
            # Calculate valid starting indices for each demo
            demo_offset = len(self.sideview_demo_list)
            for i, demo in enumerate(sideview_demos):
                self._add_valid_indices(demo, demo_idx=demo_offset + i)
            
            self.sideview_demo_list.extend(sideview_demos)
            self.task_embed_list.extend(tasks_embeds)

        self.resize = T.Resize(target_size)

        print(f"Total number of demos: {len(self.sideview_demo_list)}")
        print(f"Total number of valid sequences: {len(self.valid_indices)}")
        print("Done")

    def _add_valid_indices(self, demo, demo_idx):
        """
        Add valid starting indices for a demo to the valid_indices list.
        
        Args:
            demo: The demo sequence
            demo_idx: Index of the demo in the dataset
        """
        horizon = demo.shape[0]
        # Calculate the required sequence length
        seq_length = self.sample_per_seq + (self.sample_per_seq - 1) * self.interval
        
        # Find all valid starting indices
        for start_idx in range(horizon - seq_length + 1):
            self.valid_indices.append((demo_idx, start_idx))

    def get_demos_and_task_from_hdf5(self, hdf5_file):
        '''
        Args:
            hdf5_file: path to hdf5 file(task)
        '''
        sideview_demos = []
        with h5py.File(hdf5_file, 'r') as f:
            all_traj = list(f['data'].keys())
            num_traj = int(len(all_traj)*self.train_ratio)

            if self.mode == "train":
                train_traj = all_traj[:num_traj]
            else:
                train_traj = all_traj[-num_traj:]
                
            for traj in train_traj:
                sideview_demo = f['data'][traj]['obs']['agentview_rgb'][:]     # frames, height, width, channel
                if self.depth:
                    sideview_depth = f['data'][traj]['obs']['agentview_depth'][:]
                    sideview_demo = np.concatenate([sideview_demo,sideview_depth[:,:,:,None]], axis=3)
                sideview_demos.append(sideview_demo)
            task_embed =  f["text_embed"][0]
            tasks_embeds = [task_embed] * len(sideview_demos)

        return sideview_demos, tasks_embeds

    def get_seq_from_demo(self, demo, start_index):
        '''
        Args:
            demo: [frames, height, width, channel]
            start_index: The starting index for sampling
        Return:
            seq(np.array): [frames, height, width, channel]
        ''' 
        # Calculate end index based on interval and sample_per_seq
        end_index = start_index + self.sample_per_seq + (self.sample_per_seq - 1) * self.interval
        # Sample frames with the given interval
        seq = demo[start_index:end_index:self.interval+1]
        return seq

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        '''
        Return:
            x(torch.Tensor): [T, C, H, W]   (Normalized)
            x_cond(torch.Tensor): [C, H, W] (Normalized)
                     (We just use RGB image as condition no matter depth or not)
            task_embed(torch.Tensor): [n, D]
        '''
        # Get the demo index and start index from valid_indices
        demo_idx, start_idx = self.valid_indices[idx]
        
        # Get the demo and task embed
        sideview_demo = self.sideview_demo_list[demo_idx]
        task_embed = self.task_embed_list[demo_idx]
        task_embed = torch.from_numpy(task_embed)
        
        # Get sequence using the specific start index
        sideview_seq = self.get_seq_from_demo(sideview_demo, start_idx)
        sideview_seq = (torch.from_numpy(sideview_seq)/255.0).permute(0, 3, 1, 2)
        sideview_seq = self.resize(sideview_seq)

        x_cond = sideview_seq[0]
        x_cond = x_cond[:3]     
        
        x = sideview_seq[1:]

        return x, x_cond, task_embed
