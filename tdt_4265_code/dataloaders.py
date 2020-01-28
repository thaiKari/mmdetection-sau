#Starter code from https://github.com/hukkelas/TDT4265-A3-starter-code/blob/master/dataloaders.py

from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
from sheep_grid_dataset import SheepGridDataset

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)


def load_cifar10(batch_size, validation_fraction=0.1):
    
    transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    
    transform = transforms.Compose(transform)
    data_train = datasets.CIFAR10('data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transform)

    data_test = datasets.CIFAR10('data/cifar10',
                                 train=False,
                                 download=True,
                                 transform=transform)

    print('data_test', type(data_test))
    
    indices = list(range(len(data_train)))
    split_idx = int(np.floor(validation_fraction * len(data_train)))
    #  Uncomment to yield the same shuffle of the dataset each time
    # Note that the order of the samples will still be random, since the sampler
    # returns random indices from the list
    # np.random.seed(42)
    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2)

    dataloader_val = torch.utils.data.DataLoader(data_train,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=2)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    return dataloader_train, dataloader_val#, dataloader_test


def load_sheep_grid_data(batch_size,
                         visOnly=True,
                         root_dir = './data/data_external/',
                         labels_val_path = 'annotations/val_grid_3_3.txt',
                         image_val_path = 'val_grid_3_3',
                         labels_train_path = 'annotations/train_grid_3_3.txt',
                         image_train_path = 'train_grid_3_3',
                         grid_shape=(3,3)):
    
    data_val = SheepGridDataset(root_dir = root_dir,
                                labels_path =labels_val_path,
                                image_path =image_val_path,
                                visOnly=visOnly
                                )
    
    data_train = SheepGridDataset(root_dir = root_dir,
                                labels_path =labels_train_path,
                                 image_path =image_train_path,
                                visOnly=visOnly
                                )
    
    
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                        shuffle=True, num_workers=2)
    
    return dataloader_train, dataloader_val
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    