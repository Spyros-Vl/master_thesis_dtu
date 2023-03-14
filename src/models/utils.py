import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from cv2 import cv2
import os
import json
import time

import pathlib
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from master_thesis_dtu.src.data.my_rpg_dataset import XRayDataSet
from master_thesis_dtu.src.data.my_rpg_dataset import collate_fn
from tqdm import tqdm

#for model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.faster_rcnn import *
import torch



def get_model_instance_segmentation(num_classes):
      
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


#keep only predictions with score higher than the threshold
def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    print(preds)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : 
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds

def plot_image_from_output(img, annotation):
    
    img = img.cpu().permute(1,2,0)
    
    fig,ax = plt.subplots(1)
    ax.imshow(img,cmap='gray')
    
    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx].detach().numpy()

        if annotation['labels'][idx] == 1 :
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        
        elif annotation['labels'][idx] == 2 :
            
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='g',facecolor='none')
            
        else :
        
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='orange',facecolor='none')

        ax.add_patch(rect)

    plt.show()

def create_COCO_from_dataset(image_list,ann_list):
    
    #fix info
    info = {
            'description': 'My dataset in COCO format',
            'url': 'nan',
            'version': '1.0',
            'year': 2023,
            'contributor': 'Spyros Vlachospyros',
            'date_created': '2023-03-02'
    }
    
    #set dataset licenses
    licenses =  [
        {
            'id': 1,
            'name': 'CC-BY-4.0',
            'url': 'https://creativecommons.org/licenses/by/4.0/'
        }
    ]


    #fix the categories / it is possible to make the dataset with only one label 

    categories = [
        {
            "id": 2,
            "name": "text"
        },
        {
            "id": 1,
            "name": "fracture"
        }]
    
    anno_id = 1

    images = []
    annotations = []
    
    for idx in tqdm(range(10)):#tqdm.tqdm(range(len(img_list))):

        img_path = os.path.join(img_list[idx])
        ann_path = os.path.join(ann_list[idx])

        #fix the images format

        image_id = idx + 1
        file_name = img_path

        image = Image.open(img_path)

        # Get the width and height of the image
        width, height = image.size

        img = {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height
                }
        
        images.append(img)
            
        classes = {'fracture' : 1, 'text' : 2}


        with open(ann_path) as json_file:
                
            #Load the JSON file
            data = json.load(json_file)


            for object_dict in data['objects']:
            
                # Check if object contains any fractures

                if object_dict['classTitle'] == "text" or object_dict['classTitle'] == "fracture":

                    annotation = {}
                    box = []
                    area = []
                    coco_bbox = []

                    # Get points and convert them to int for display purposes
                    top_left_point, bottom_right_point = object_dict['points']['exterior']
                    top_left_point = list(map(int,top_left_point))
                    bottom_right_point = list(map(int, bottom_right_point))
                    box = (top_left_point+bottom_right_point)
                    #turn boxex from xmin,ymin,xmax,ymax format to coco format x,y,w,h
                    x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                    coco_bbox = [x, y, w, h] 
                    area.append((box[3] - box[1]) * (box[2] - box[0]))
                    label = classes[object_dict['classTitle']]
                    annotation["id"] = anno_id
                    annotation["image_id"] = image_id
                    annotation["category_id"] = label
                    annotation["bbox"] = coco_bbox
                    annotation["area"] = (box[3] - box[1]) * (box[2] - box[0])
                    annotation["segmentation"] = []
                    annotation["iscrowd"] = 0

                    anno_id += 1
                    annotations.append(annotation)


    COCO_dataset = {
        "info" : info,
        "licenses" : licenses,
        "categories" : categories,
        "images" : images,
        "annotations" : annotations
    }

    # Write COCO data to JSON file
    with open('coco_data.json', 'w') as f:
        json.dump(COCO_dataset, f)

    return COCO_dataset

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results