import numpy as np
import cv2
import torch
from torch.utils import data
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import os
import math

class Dataset(data.Dataset):
    def __init__(self, list_IDs, labels, cfg):
        'Initialization'
        
        self.list_IDs = list_IDs
        self.labels = labels
        self.cfg = cfg
            
        self.normalize = transforms.Normalize(cfg['dataset_params']['mean'], cfg['dataset_params']['std'])
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        im_size = self.cfg['dataset_params']['im_size']
        image = cv2.imread(self.list_IDs[index],1)
        image = cv2.resize(image, (im_size,im_size))
        image = image/255.0
        image = np.moveaxis(image, 2, 0)
        image = np.float32(image)
        image = torch.from_numpy(image)
        image = self.normalize(image)

        mask = cv2.imread(self.labels[index],0)
        mask = cv2.resize(mask, (im_size,im_size), interpolation = cv2.INTER_NEAREST)
        mask = mask/255.0
        mask = torch.from_numpy(mask)
        return image, mask

def get_file_names(cfg):

    train_IDs = {}
    mask_train_IDs = {}
    val_IDs = {}
    mask_val_IDs = {}
    test_IDs = {}
    mask_test_IDs = {}

    with open('dataloader/train_imd.txt', 'r') as file:
        train_files = file.read().split('\n')
    with open('dataloader/validation_imd.txt', 'r') as file:
        val_files = file.read().split('\n')
    with open('dataloader/test_imd.txt', 'r') as file:
        test_files = file.read().split('\n')
    
    for i in range(len(train_files)):
        train_IDs[i] = os.path.join(cfg['dataset_params']['base_dir'], train_files[i])
        mask_file_name = train_files[i].split('_0.jpg')[0]+'_0_mask.png'
        mask_train_IDs[i] = os.path.join(cfg['dataset_params']['base_dir'], mask_file_name)
    
    for i in range(len(val_files)):
        val_IDs[i] = os.path.join(cfg['dataset_params']['base_dir'], val_files[i])
        mask_file_name = val_files[i].split('_0.jpg')[0]+'_0_mask.png'
        mask_val_IDs[i] = os.path.join(cfg['dataset_params']['base_dir'], mask_file_name)

    for i in range(len(test_files)):
        test_IDs[i] = os.path.join(cfg['dataset_params']['base_dir'], test_files[i])
        mask_file_name = test_files[i].split('_0.jpg')[0]+'_0_mask.png'
        mask_test_IDs[i] = os.path.join(cfg['dataset_params']['base_dir'], mask_file_name)
    
    
    return train_IDs, mask_train_IDs, val_IDs, mask_val_IDs, test_IDs, mask_test_IDs

class generator():
    def __init__(self,cfg):
        self.cfg = cfg
        self.train_IDs, self.mask_train_IDs, self.val_IDs, self.mask_val_IDs, self.test_IDs, self.mask_test_IDs = get_file_names(cfg)
    

    def get_train_generator(self):
        
        batch_size = self.cfg['dataset_params']['batch_size']
        params = {'batch_size': batch_size,
                    'shuffle': True,
                    'pin_memory':True,
                    'num_workers': 4}
        training_set = Dataset(self.train_IDs, self.mask_train_IDs, self.cfg)
        training_generator = data.DataLoader(training_set, **params)

        return training_generator

    def get_val_generator(self):

        batch_size = self.cfg['dataset_params']['batch_size']
        params = {'batch_size': batch_size,
                    'shuffle': False,
                    'pin_memory':True,
                    'num_workers': 4}
        val_set = Dataset(self.val_IDs, self.mask_val_IDs, self.cfg)
        validation_generator = data.DataLoader(val_set, **params)
        return validation_generator
    
    def get_test_generator(self):

        batch_size = self.cfg['dataset_params']['batch_size']
        params = {'batch_size': batch_size,
                    'shuffle': False,
                    'pin_memory':True,
                    'num_workers': 4}
        test_set = Dataset(self.test_IDs, self.mask_test_IDs, self.cfg)
        test_generator = data.DataLoader(test_set, **params)
        return test_generator

