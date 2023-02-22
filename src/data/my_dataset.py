import torch
import pathlib
from torchvision.ops import box_convert
from typing import List, Dict
from skimage.io import imread
import os
from PIL import Image
import numpy as np
import json


class XRayDataSet(torch.utils.data.Dataset):


     def __init__(self, root):
        self.root = root

        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        self.anno = list(sorted(os.listdir(os.path.join(root, "ann"))))

    
     def __len__(self):
        return len(self.imgs) 

     def __getitem__(self,idx):
        
        img_path = os.path.join(self.root, "img", self.imgs[idx])
        ann_path = os.path.join(self.root, "ann", self.anno[idx])

        # Load input and target
        img = Image.open(img_path)

        obj_ids = np.unique(img)

        # Load json
        data = json.load(ann_path)

        num_objs = len(obj_ids)
       
        
        for object_dict in data['objects']:
            # Check if object contains any fractures 

            box = []
            boxes = []
            labels = []

            if object_dict['classTitle'] == "fracture":
                # Get points and convert them to int for display purposes
                top_left_point, bottom_right_point = object_dict['points']['exterior']
                top_left_point = list(map(int,top_left_point))
                bottom_right_point = list(map(int, bottom_right_point))
                box = (top_left_point+bottom_right_point)
                label = 1
                labels.append(torch.as_tensor(label, dtype=torch.int642))
                boxes.append(torch.FloatTensor(box))
        
            if not boxes:
                label = 2
                box = torch.zeros(4)
                labels.append(torch.as_tensor(label, dtype=torch.int642))
                boxes.append(torch.FloatTensor(box))



        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd


        return img, target

     def read_images(inp, tar):
        return imread(inp), torch.load(tar)

    