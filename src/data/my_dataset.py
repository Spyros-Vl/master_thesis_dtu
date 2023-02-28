import torch
import pathlib
from torchvision.ops import box_convert
from typing import List, Dict
from skimage.io import imread
import os
from PIL import Image
import numpy as np
import json
import pickle


class XRayDataSet(torch.utils.data.Dataset):


     def __init__(self, root):
        
        self.root = root
        self.instances = list(sorted(os.listdir(root)))
        

    
     def __len__(self):
        return len(self.instances) 

     def __getitem__(self,idx):
        
        instance = os.path.join(self.root, self.instances[idx])
        

        with open(instance, 'rb') as handle:
            data = pickle.load(handle)

        img = data['image']
        target = data['target']


        return img, target

     def read_images(inp, tar):
        return imread(inp), torch.load(tar)


def collate_fn(batch):
    return tuple(zip(*batch))