from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import read_grid_labels

class SheepGridDataset(Dataset):

    def __init__(self, labels_path, image_path, root_dir, visOnly, transform=None, grid_shape=(3,3)):

        self.labels = read_grid_labels(os.path.join(root_dir,labels_path))
        self.image_path = image_path
        self.root_dir = root_dir
        self.transform = transform
        self.grid_shape = grid_shape
        self.visOnly = visOnly

    
    def get_keys(self):
        return list(self.labels.keys())
    
    def get_grid_shape(self):
        return self.grid_shape    
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        key = list(self.labels.keys())[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_full_path = os.path.join(self.root_dir,
                                self.image_path, key + '.npy')
        image = np.load(img_full_path)
        if self.visOnly:
            image = image[:,:,:3]
        image = torch.from_numpy(image).permute(2,0,1)
        image = image.float()
        label = self.labels[key]
        label = torch.from_numpy(label)
        sample = (image, label, key)

        if self.transform:
            sample = self.transform(sample)

        return sample