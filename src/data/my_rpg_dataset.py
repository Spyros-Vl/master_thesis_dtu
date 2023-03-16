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
import torchvision
from transformers import DetrImageProcessor
import torchvision.transforms as T
from cv2 import cv2


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

        img_path = data['image']
        linux_path = os.path.join(*img_path.split('\\'))
        #for linux to work
        start_dir = '../'
        img_path = os.path.relpath(linux_path, start_dir)
        #

        img = cv2.imread(img_path)
        img = T.ToTensor()(img).float()
        target = data['target']


        return img, target

     def read_images(inp, tar):
        return imread(inp), torch.load(tar)



class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, path_folder, processor, train=True):
        ann_file = os.path.join(path_folder, "test_coco_data.json" if train else "test_coco_data.json")
        super(CocoDetection, self).__init__(path_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target


def collate_fn_COCO(batch):

   checkpoint = "facebook/detr-resnet-50"    
   processor = DetrImageProcessor.from_pretrained(checkpoint)

   pixel_values = [item[0] for item in batch]
   encoding = processor.pad(pixel_values, return_tensors="pt")
   labels = [item[1] for item in batch]
   batch = {}
   batch['pixel_values'] = encoding['pixel_values']
   batch['pixel_mask'] = encoding['pixel_mask']
   batch['labels'] = labels
   return batch

def collate_fn(batch):
    return tuple(zip(*batch))
