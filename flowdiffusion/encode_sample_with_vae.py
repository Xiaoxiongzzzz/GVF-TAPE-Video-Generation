import h5py
import torch
import os
import cv2
from tqdm import tqdm
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from torchvision import transforms
from glob import glob

FOLDER_PATH_LIST = [
    "/mnt/home/ZhangXiaoxiong/Data/final_tasks/put_the_blue_bowl_on_the_red_plate",
    "/mnt/home/ZhangXiaoxiong/Data/final_tasks/put_the_green_cup_on_the_pink_plate",
    "/mnt/home/ZhangXiaoxiong/Data/final_tasks/put_the_red_cup_on_the_silver_plate",
    "/mnt/home/ZhangXiaoxiong/Data/final_tasks/put_the_sponge_on_the_red_plate",
    ]
OUTPUT_ROOT = "/mnt/home/ZhangXiaoxiong/Data/real-data-2"

device = torch.device("cuda")
batch_size = 32
origin_size = (480, 640)
latent_size = tuple(dim // 8 for dim in origin_size)

transform = transforms.Compose([
    transforms.ToTensor(),
    ])
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)

for folder_path in FOLDER_PATH_LIST:
    print(f"Processing folder: {folder_path}")
    hdf5_file_list = glob(os.path.join(folder_path, "*.hdf5"), recursive=True)
    
    for hdf5_path in tqdm(hdf5_file_list):
        rel_path = os.path.relpath(hdf5_path, os.path.dirname(folder_path))
        output_path = os.path.join(OUTPUT_ROOT, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with h5py.File(hdf5_path, 'r') as hdf5_file:

            with h5py.File(output_path, 'w') as dst_file:
                #  copy original data
                for key in hdf5_file.keys():
                    hdf5_file.copy(key, dst_file)

                image_data = hdf5_file['observations']["images"]["mid"][:]  
                image_data = torch.stack([transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in image_data]).to(device)

                num_batches = len(image_data) // batch_size + (len(image_data) % batch_size > 0)

                if 'latent_observation' in dst_file.keys():
                    del dst_file['latent_observation']

                latent_shape = (len(image_data), vae.config.latent_channels, latent_size[0], latent_size[1])
                latent_dataset = dst_file.create_dataset('latent_observation', shape=latent_shape, dtype='float32')

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(image_data))
                    
                    batch_images = image_data[start_idx:end_idx]
                    
                    with torch.no_grad():
                        latent = vae.encode(batch_images, return_dict=True).latent_dist.mean
                        latent_dataset[start_idx:end_idx] = latent.cpu().numpy()