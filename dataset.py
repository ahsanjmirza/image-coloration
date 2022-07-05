import torch
from torch.utils.data import Dataset
from imageio import imread
import numpy as np
from skimage import transform, color
import os

class VintageFacesDataset(Dataset):
    def __init__(self, img_dir, size):
        self.img_dir = img_dir
        self.size = size
        self.img_paths = []
        for i in os.listdir(self.img_dir):
            self.img_paths.append(os.path.join(self.img_dir, i))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Read image
        rgb = np.float32(imread(self.img_paths[idx]))
        if len(rgb.shape) != 3:
            rgb = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8)

        # Resize the image to new size
        rgb = transform.resize(
                image=rgb[:, :, :3], 
                output_shape=(self.size, self.size),
                order=1,
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            )
        
        # Clip to original range
        rgb = np.uint8(np.clip(rgb, 0, 255))

        # Convert rgb to ycbcr
        lab = color.rgb2lab(rgb)

        L = np.zeros((1, self.size, self.size), dtype=np.float32)
        ab = np.zeros((2, self.size, self.size), dtype=np.float32)
        L[0] = lab[:, :, 0] / 100
        ab[0] = lab[:, :, 1]
        ab[1] = lab[:, :, 2]
        
        return torch.from_numpy(L), torch.from_numpy(ab)



        