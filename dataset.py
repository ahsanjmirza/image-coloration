import torch
from torch.utils.data import Dataset
from imageio import imread
import numpy as np
from skimage import color, draw
import os
import random
from matplotlib import pyplot as plt

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
        rgb = imread(self.img_paths[idx])
        
        if len(rgb.shape) != 3:
            rgb = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8)

        lab = color.rgb2lab(rgb)

        L = np.zeros((1, self.size, self.size), dtype=np.float32)
        lab_new = np.zeros((3, self.size, self.size), dtype=np.float32)
    
        rnd = np.random.randint(low=0, high=100, size=(self.size, self.size), dtype=np.uint8)
        L[0] = np.where(rnd > 98, 100, lab[:, :, 0])
        L[0] = self.add_scratches(L[0], (20, 120), (2, 6))

        plt.imshow(L[0] / 100, 'Greys_r')
        plt.show()

        L[0] = (L[0] / 50) - 1

        lab_new[0] = (lab[:, :, 0] / 50) - 1
        lab_new[1] = lab[:, :, 1] / 128
        lab_new[2] = lab[:, :, 2] / 128
        
        return torch.from_numpy(L), torch.from_numpy(lab_new)

    def back_transform(self, from_nn):
        lab_reshaped = from_nn.detach().cpu().numpy()
        lab_new = np.zeros((256, 256, 3), dtype=np.float32)
        lab_new[:, :, 0] = np.clip((lab_reshaped[0, 0, :, :] + 1) * 50, 0, 100)
        lab_new[:, :, 1] = np.clip(lab_reshaped[0, 1, :, :] * 128, -127, 127)
        lab_new[:, :, 2] = np.clip(lab_reshaped[0, 2, :, :] * 128, -127, 127)
        rgb_new = np.clip(color.lab2rgb(lab_new) * 255, 0, 255)
        return np.uint8(rgb_new)

    def add_scratches(self, img, no_scratches, scratch_length):
        s0, s1 = img.shape[0]-1, img.shape[1]-1
        for _ in range(random.randint(no_scratches[0], no_scratches[1])):
            length_s0, length_s1 = random.randint(-scratch_length[0], scratch_length[1]), random.randint(-scratch_length[0], scratch_length[1])
            origin = (random.randint(0, s0), random.randint(0, s1))
            r0, r1 = origin[0], origin[0]+length_s0
            c0, c1 = origin[1], origin[1]+length_s1
            r1 = 0 if r1 < 0 else r1
            c1 = 0 if c1 < 0 else c1
            r1 = s0 if r1 > s0 else r1 
            c1 = s1 if c1 > s1 else c1
            rr, cc = draw.line(r0, c0, r1, c1)
            img[rr, cc] = 100
        return img



        