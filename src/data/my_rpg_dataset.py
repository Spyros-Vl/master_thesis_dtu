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
import cv2


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

class XRayDataSet_aug(torch.utils.data.Dataset):
    

     def __init__(self, root,transform):
         
        self.root = root
        self.instances = list(sorted(os.listdir(root)))
        self.transform = transform
        

    
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
        target = data['target']

        transformed = self.transform(image=img, bboxes=target['boxes'].detach().cpu().numpy(), class_labels=target['labels'].detach().cpu().numpy())

        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']

        d = {}

        boxes = torch.FloatTensor(transformed_bboxes)
        labels = torch.as_tensor(transformed_class_labels, dtype=torch.int64)
                
        
        d["boxes"] = boxes
        d["labels"] = labels
        d["image_id"] = target["image_id"]
        d["area"] = target['area']

        

        target = d


        img = T.ToTensor()(transformed_image).float()


        return img, target
     
class XRayDataSet_windows(torch.utils.data.Dataset):
    

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

         img = cv2.imread(img_path)
         img = T.ToTensor()(img).float()
         target = data['target']

         


         return img, target
      
class XRayDataSet_windows_transform(torch.utils.data.Dataset):
    

      def __init__(self, root,transform):
         
         self.root = root
         self.instances = list(sorted(os.listdir(root)))
         self.transform = transform
         
         

      
      def __len__(self):
         return len(self.instances) 

      def __getitem__(self,idx):
         
        instance = os.path.join(self.root, self.instances[idx])
         

        with open(instance, 'rb') as handle:
            data = pickle.load(handle)

        img_path = data['image']
        linux_path = os.path.join(*img_path.split('\\'))

        img = cv2.imread(img_path)
        target = data['target']
         
        transformed = self.transform(image=img, bboxes=target['boxes'].detach().cpu().numpy(), class_labels=target['labels'].detach().cpu().numpy())

        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']

        d = {}

        boxes = torch.FloatTensor(transformed_bboxes)
        labels = torch.as_tensor(transformed_class_labels, dtype=torch.int64)
                
        
        d["boxes"] = boxes
        d["labels"] = labels
        d["image_id"] = target["image_id"]
        d["area"] = target['area']

        

        target = d


        img = T.ToTensor()(transformed_image).float()


        return img, target
      
class XRayDataSet_coco(torch.utils.data.Dataset):
    

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
      
      
      target = data['target']


      return img_path, target

def read_images(inp, tar):
   return imread(inp), torch.load(tar)



class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, path_folder, processor, status="train"):
        if status == "train":
            ann_file = os.path.join(path_folder, "train_coco_data.json")
        elif status == "validation":
            ann_file = os.path.join(path_folder, "val_coco_data.json")
        elif status == "test":
            ann_file = os.path.join(path_folder, "test_coco_data.json")
        else:
            raise ValueError("Invalid value for status. Expected 'train', 'validation', or 'test'.")
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
    
class MultiViewCocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, path_folder, processor, status="train"):
        if status == "train":
            ann_file = os.path.join(path_folder, "train_multi_coco_data.json")
        elif status == "validation":
            ann_file = os.path.join(path_folder, "val_multi_coco_data.json")
        elif status == "test":
            ann_file = os.path.join(path_folder, "test_multi_coco_data.json")
        else:
            raise ValueError("Invalid value for status. Expected 'train', 'validation', or 'test'.")
        super(MultiViewCocoDetection, self).__init__(path_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(MultiViewCocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

class CocoDetection_aug(torchvision.datasets.CocoDetection):
    def __init__(self, path_folder, processor, transform, status="train"):
        if status == "train":
            ann_file = os.path.join(path_folder, "train_coco_data.json")
        elif status == "validation":
            ann_file = os.path.join(path_folder, "val_coco_data.json")
        elif status == "test":
            ann_file = os.path.join(path_folder, "test_coco_data.json")
        else:
            raise ValueError("Invalid value for status. Expected 'train', 'validation', or 'test'.")
        super(CocoDetection_aug, self).__init__(path_folder, ann_file)
        self.processor = processor
        self.transform = transform

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection_aug, self).__getitem__(idx)

        img = np.array(img)

        # Extract bounding boxes and labels from the annotations
        bboxes = []
        class_labels = []
        for annotation in target:
            bboxes.append(annotation['bbox'])
            class_labels.append(annotation['category_id'])

        # Perform augmentation
        transformed= self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
        augmented_image = transformed['image']
        augmented_bboxes = transformed['bboxes']
        augmented_labels = transformed['class_labels']

        img = Image.fromarray(augmented_image)

        # Update the target with augmented bounding boxes and labels
        augmented_target = []
        for bbox, label in zip(augmented_bboxes, augmented_labels):
            augmented_annotation = {
                'id': len(augmented_target),
                'image_id': target[0]['image_id'],
                'category_id': label,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'segmentation': [],
                'iscrowd': 0
            }
            augmented_target.append(augmented_annotation)

        # Preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': augmented_target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

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
