#Starter code from https://github.com/hukkelas/TDT4265-A3-starter-code/blob/master/dataloaders.py

from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
from sheep_grid_dataset import SheepGridDataset, SheepGridDatasetMultiBand


def load_sheep_grid_data(batch_size,
                         img_type='rgb',
                         include_msx = False,
                         root_dir = './data/data_external/',
                         labels_val_path = 'annotations/val2020_simple.json',
                         image_val_path = 'val2020',
                         labels_train_path = 'annotations/train2020_simple.json',
                         image_train_path = 'train2020',
                         grid_shape=(3,3),
                         test_mode = False,
                         rgb_resize_shape = (1280,1280),
                         infrared_resize_shape = (160,160)):
    
  
    data_val = SheepGridDataset(root_dir = root_dir,
                                labels_path =labels_val_path,
                                image_path =image_val_path,
                                img_type=img_type,
                                include_msx = include_msx,
                                test_mode=True, #Validation set is always in test mode
                                rgb_resize_shape=rgb_resize_shape,
                                infrared_resize_shape=infrared_resize_shape,                                
                                )
    
    data_train = SheepGridDataset(root_dir = root_dir,
                                labels_path =labels_train_path,
                                image_path =image_train_path,
                                img_type=img_type,
                                include_msx = include_msx,
                                test_mode=test_mode,
                                rgb_resize_shape=rgb_resize_shape,
                                infrared_resize_shape=infrared_resize_shape
                                )
    

    
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    
    
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                        shuffle=True, num_workers=2)
    
    return dataloader_train, dataloader_val
    
    
   
    
    
def load_sheep_grid_multiband(batch_size,
                         root_dir = './data/data_external/',
                         labels_val_path = 'annotations/val2020_simple.json',
                         image_val_path = 'val2020',
                         labels_train_path = 'annotations/train2020_simple.json',
                         image_train_path = 'train2020',
                         grid_shape=(3,3),
                         test_mode = False,
                         rgb_resize_shape = (1280,1280),
                         infrared_resize_shape = (160,160)
                             ):
    
    data_val = SheepGridDatasetMultiBand(root_dir = root_dir,
                                labels_path =labels_val_path,
                                image_path =image_val_path,
                                test_mode=True,
                                rgb_resize_shape=rgb_resize_shape,
                                infrared_resize_shape=infrared_resize_shape
                                )
    
    data_train = SheepGridDatasetMultiBand(root_dir = root_dir,
                                labels_path =labels_train_path,
                                 image_path =image_train_path,
                                test_mode=test_mode,
                                rgb_resize_shape=rgb_resize_shape,
                                infrared_resize_shape=infrared_resize_shape
                                )
    

    
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    
    
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                        shuffle=(not test_mode), num_workers=2)
    

    return dataloader_train, dataloader_val    
   
    
    
    
'''
 
    
def load_sheep_grid_data_test(batch_size,
                         img_type='rgb',
                         root_dir = './data/data_external/',
                         labels_val_path = 'annotations/val2020_simple.json',
                         image_val_path = 'val2020',
                         labels_train_path = 'annotations/train2020_simple.json',
                         image_train_path = 'train2020',
                         grid_shape=(3,3)):
    
    
    if img_type == 'ensemble':
        data_val = SheepGridDatasetMultiBand(root_dir = root_dir,
                                labels_path =labels_val_path,
                                image_path =image_val_path,
                                test_mode = True
                                )
    
        data_train = SheepGridDatasetMultiBand(root_dir = root_dir,
                                    labels_path =labels_train_path,
                                    image_path =image_train_path,
                                    test_mode = True                                               
                                    )
    else:    
        data_val = SheepGridDatasetTest(root_dir = root_dir,
                                    labels_path =labels_val_path,
                                    image_path =image_val_path,
                                    img_type=img_type
                                    )

        data_train = SheepGridDatasetTest(root_dir = root_dir,
                                    labels_path =labels_train_path,
                                     image_path =image_train_path,
                                    img_type=img_type
                                    )
    

    
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    
    
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    
    return dataloader_train, dataloader_val
    
'''
    
    
    
    
    
    
    
    