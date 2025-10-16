import os

from PIL import Image
from torch.utils.data import Dataset
import torch


class LatentsData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.latents_dir = os.listdir(self.root_dir)


    def __len__(self):
        return len(self.latents_dir)


    def __getitem__(self, idx):
        latents_path = os.path.join(self.root_dir, self.latents_dir[idx])
        gt_latents_path = os.path.join(latents_path,'gt_latents.pt')
        org_latents_path = os.path.join(latents_path, 'org_latents.pt')
        image_latents_path = os.path.join(latents_path, 'image_latenets_without_noise.pt')
        gt_latents = torch.load(gt_latents_path)
        org_latents = torch.load(org_latents_path)
        image_latents = torch.load(image_latents_path)

        return gt_latents, org_latents,image_latents

