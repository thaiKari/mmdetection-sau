from __future__ import print_function, division
import os
import torch
import io
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import (read_coco_annotations,
                   read_coco_annotations_test,
                   get_image_crop,
                    get_grid)
from transforms import (common_augmentations,
                    rgb_augmentations,
                    infrared_augmentations,
                   rgb_augmentations_bare_bones,
                      )
from albumentations import ReplayCompose


class SheepGridDataset(Dataset):
    #img_type = rgb or infrared
    def __init__(self,
                 labels_path,
                 image_path,
                 root_dir,
                 img_type='rgb',
                 crop_shape=(1200,1200),
                 im_shape=(2400,3200),
                 grid_shape=(3,3),
                 include_msx = False,
                 test_mode = False,
                 rgb_resize_shape = (1280,1280),
                 infrared_resize_shape = (160,160)):

        if test_mode:
            self.labels = read_coco_annotations_test(os.path.join(root_dir,labels_path), crop_shape=crop_shape, im_shape=im_shape, grid_shape=grid_shape)
        else:    
            self.labels = read_coco_annotations(os.path.join(root_dir,labels_path))
        
        self.image_path = image_path
        self.root_dir = root_dir
        self.grid_shape = grid_shape
        self.test_mode = test_mode
        self.img_type = img_type
        self.include_msx = include_msx
        self.crop_shape = crop_shape
        self.im_shape = im_shape
        self.rgb_resize_shape = rgb_resize_shape
        self.infrared_resize_shape = infrared_resize_shape

    
    def get_keys(self):
        return list(self.labels.keys())
    
    def get_grid_shape(self):
        return self.grid_shape    
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        key = list(self.labels.keys())[idx]
        label = self.labels[key]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.test_mode:
            im_key = key.split('.JPG')[0]+ '.JPG'
        else:
            im_key = key

        img_full_path = os.path.join(self.root_dir,
                            self.image_path, self.img_type ,im_key)
        
        if self.include_msx: #img could be in msx folder
            if self.img_type =='rgb':
                im_folder_name = 'rgb_msx'
                msx_im_list = os.listdir(os.path.join(self.root_dir,
                            self.image_path, im_folder_name))
            elif self.img_type =='infrared':
                im_folder_name = 'msx'
                msx_im_list = os.listdir(os.path.join(self.root_dir,
                            self.image_path, im_folder_name))
                
            if im_key in msx_im_list:
                img_full_path = os.path.join(self.root_dir,
                            self.image_path, im_folder_name ,im_key)
        
        image = io.imread(img_full_path)
        
        if self.test_mode:
            image = get_image_crop(image, key, crop_shape=self.crop_shape, im_shape=self.im_shape, grid_shape=self.grid_shape)
        
        sample = { 'image': image,
                  'bboxes': label,
                  'category_id': [0]*len(label) }
        
        #TRANSFORMS
        if not self.test_mode:
            sample = common_augmentations()(**sample)
        
        if self.img_type == 'rgb':
            if  self.test_mode:
                sample = rgb_augmentations_bare_bones(resize_shape=self.rgb_resize_shape)(**sample)
            else:
                sample = rgb_augmentations(resize_shape=self.rgb_resize_shape)(**sample)
        elif self.img_type == 'infrared':
            sample = infrared_augmentations(resize_shape=self.infrared_resize_shape)(**sample)
            
            
        # CALCULATE GRID VALUES FROM BBOXES
        sample['grid'] = get_grid(sample['bboxes'], sample['image'].shape, self.grid_shape)
        
        #TO TORCH
        sample['image'] = torch.from_numpy(sample['image']).permute(2,0,1)
        #sample['bboxes'] = torch.from_list(sample['bboxes'])
        sample['grid'] = torch.from_numpy(sample['grid'])


        return {'image': sample['image'],
                'grid': sample['grid'],
                'key': key}
    
class SheepGridDatasetMultiBand(Dataset):

    def __init__(self,root_dir,
                 labels_path,
                 image_path,
                 crop_shape=(1200,1200),
                 im_shape=(2400,3200),
                 grid_shape=(3,3),
                 test_mode=False,
                 rgb_resize_shape = (1280,1280),
                 infrared_resize_shape = (160,160)):

        
        if test_mode:
            self.labels = read_coco_annotations_test(os.path.join(root_dir,labels_path), crop_shape=crop_shape, im_shape=im_shape, grid_shape=grid_shape)
        else:
            self.labels = read_coco_annotations(os.path.join(root_dir,labels_path))
        self.image_path = image_path
        self.root_dir = root_dir
        self.transform = transform
        self.grid_shape = grid_shape
        self.test_mode = test_mode
        self.im_shape=im_shape
        self.crop_shape = crop_shape
        self.rgb_resize_shape = rgb_resize_shape
        self.infrared_resize_shape = infrared_resize_shape
        

    
    def get_keys(self):
        return list(self.labels.keys())
    
    def get_grid_shape(self):
        return self.grid_shape    
    
         
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        key = list(self.labels.keys())[idx]
        label = self.labels[key]

        if torch.is_tensor(idx):
            idx = idx.tolist()
            
            
        if self.test_mode:
            im_key = key.split('.JPG')[0]+ '.JPG'
        else:
            im_key = key
        
        img_rgb_full_path = os.path.join(self.root_dir,
                                self.image_path, 'rgb',im_key)
        image_rgb = io.imread(img_rgb_full_path)
        
        img_infrared_full_path = os.path.join(self.root_dir,
                                self.image_path, 'infrared',im_key)
        image_infrared = io.imread(img_infrared_full_path)
        
        
        if self.test_mode:
            image_rgb = get_image_crop(image_rgb, key, crop_shape=self.crop_shape, im_shape=self.im_shape, grid_shape=self.grid_shape)
            image_infrared = get_image_crop(image_infrared, key, crop_shape=self.crop_shape, im_shape=self.im_shape, grid_shape=self.grid_shape)

        
        
        sample = {'rgb': {'image': image_rgb,
                         'bboxes': label,
                         'category_id': [0]*len(label)},
                  'infrared': {'image': image_infrared,
                         'bboxes': label,
                         'category_id': [0]*len(label)},
                  'bboxes': label,
                  'category_id': [0]*len(label),
                  'key': key}

        # TRANSFORMS
        if  self.test_mode:
            transformed_rgb = rgb_augmentations_bare_bones(resize_shape=self.rgb_resize_shape)(**sample['rgb'])   
            transformed_infrared = infrared_augmentations(resize_shape=self.infrared_resize_shape)(**sample['infrared'])
            
        
        else:
            transformed_rgb = common_augmentations()(**sample['rgb'])      
            transformed_infrared_im = ReplayCompose.replay(transformed_rgb['replay'], image=sample['infrared']['image'])['image']
            transformed_infrared = transformed_rgb.copy()
            transformed_infrared['image'] = transformed_infrared_im
        
            transformed_rgb = rgb_augmentations(resize_shape=self.rgb_resize_shape)(**transformed_rgb)
            transformed_infrared= infrared_augmentations(resize_shape=self.infrared_resize_shape)(**transformed_infrared)
         
        #print('min', np.min( transformed_infrared['image']), 'max', np.max( transformed_infrared['image']))
        #print('min_rgb', np.min( transformed_rgb['image']), 'max_rgb', np.max( transformed_rgb['image']))
        #print()
        

#sample['image'] = torch.from_numpy(sample['image']).permute(2,0,1)

        return {'rgb': torch.from_numpy(transformed_rgb['image']).permute(2,0,1),
                       'infrared': torch.from_numpy(transformed_infrared['image']).permute(2,0,1),
                        'grid': torch.from_numpy(get_grid(transformed_rgb['bboxes'], transformed_rgb['image'].shape, self.grid_shape)),
                        'key': key}



'''

## TEST: READ CROPS COVERING FULL IMAGE. NO FANCY AUGMENTATION
class SheepGridDatasetTest(Dataset):
    #img_type = rgb or infrared
    def __init__(self, labels_path, image_path, root_dir, img_type='rgb', crop_shape=(1200,1200), im_shape=(2400,3200), grid_shape=(3,3), include_msx = False):

        self.labels = read_coco_annotations_test(os.path.join(root_dir,labels_path), crop_shape=crop_shape, im_shape=im_shape, grid_shape=grid_shape)
        self.image_path = image_path
        self.root_dir = root_dir
        self.crop_shape = crop_shape
        self.im_shape = im_shape
        self.grid_shape = grid_shape        
        self.img_type = img_type
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.include_msx = include_msx

    
    def get_keys(self):
        return list(self.labels.keys())
    
    def get_grid_shape(self):
        return self.grid_shape    
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        key = list(self.labels.keys())[idx]
        label = self.labels[key]
        
        if torch.is_tensor(idx):
            idx = idx.tolist()


            
        #key format: fullimkey_gridminx_gridminy -> need to extract correct crop of image
        im_key = key.split('.JPG')[0]+ '.JPG'
        
        img_full_path = os.path.join(self.root_dir,
                            self.image_path, self.img_type ,im_key)
        
        
        if self.include_msx: #img could be in msx folder
            if img_type =='rgb':
                im_folder_name = 'rgb_msx'
                msx_im_list = os.lisdir(os.path.join(self.root_dir,
                            self.image_path, im_folder_name ,im_key))
            elif img_type =='infrared':
                im_folder_name = 'msx'
                msx_im_list = os.lisdir(os.path.join(self.root_dir,
                            self.image_path, im_folder_name ,im_key))
                
            if im_key in msx_im_list:
                img_full_path = os.path.join(self.root_dir,
                            self.image_path, im_folder_name ,im_key)
            
        
        
        image = io.imread(img_full_path)
        image = get_image_crop(image, key, crop_shape=self.crop_shape, im_shape=self.im_shape, grid_shape=self.grid_shape)
        
        
        
 
        sample = { 'image': image,
                  'bboxes': label,
                  'category_id': [0]*len(label) }
        
        #TRANSFORMS (just resize and normalize)
        if self.img_type == 'rgb':
            sample = rgb_augmentations_bare_bones()(**sample)
        elif self.img_type == 'infrared':
            sample = infrared_augmentations()(**sample)
            
            
        # CALCULATE GRID VALUES FROM BBOXES
        sample['grid'] = get_grid(sample['bboxes'], sample['image'].shape, self.grid_shape)
        
        #TO TORCH
        sample['image'] = torch.from_numpy(sample['image']).permute(2,0,1)
        sample['bboxes'] = torch.from_numpy(np.array(sample['bboxes']))
        sample['grid'] = torch.from_numpy(sample['grid'])
        
        
        
        #TO TORCH
        #sample['image'] = self.transform(sample['image'])#.permute(2,0,1)
        #sample['bboxes'] = torch.from_list(sample['bboxes'])
        #sample['grid'] = torch.from_numpy(sample['grid'])

        return {'image': sample['image'],
                'grid': sample['grid'],
                #'bboxes': sample['bboxes'],
                'key': key}
'''