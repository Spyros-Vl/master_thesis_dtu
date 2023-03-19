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
import torch
from utils import *
import numpy as np
from torchvision.ops import box_iou

import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from src.data.my_rpg_dataset import CocoDetection
from src.data.my_rpg_dataset import collate_fn_COCO
from src.data.my_rpg_dataset import XRayDataSet_coco


def main():

    # train on the GPU or on the CPU, if a GPU is not available
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    #defines
    BatchSize = 1
    num_workers =4

    score_threshold = 0.8
    iou_threshold = 0.5

    #load test data
    test_dataset = XRayDataSet(pathlib.Path('literature/Other/supervisely/wrist/test_pickles'))
    test_dataloader = DataLoader(test_dataset, batch_size=BatchSize, shuffle=False, num_workers=num_workers,collate_fn=collate_fn)



    #load the model state
    model = get_model_instance_segmentation(3)
    model.load_state_dict(torch.load(f'CNN_Model.pt'))
    model.to(device)

    # Create a dictionary to hold the converted annotations
    coco_annotations = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'fracture'}, {'id': 2, 'name': 'text'}]
    }

    # Convert your annotations to COCO format
    for i, (img_path, targets) in enumerate(test_dataset):
        img_id = i + 1
        
        file_name = img_path

        # Load the image using cv2
        image = cv2.imread(img_path)

        # Get the height and width of the image
        height, width, channels = image.shape

        coco_annotations['images'].append({
            'id': img_id,
            'file_name': file_name.replace("\\","/"),
            'height': height,
            'width': width
        })
        for j in range(len(targets['boxes'])):
            #turn boxex from xmin,ymin,xmax,ymax format to coco format x,y,w,h
            box = targets['boxes'][j]
            x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1] 
            area = (box[3] - box[1]) * (box[2] - box[0])
            coco_annotations['annotations'].append({
                'id': len(coco_annotations['annotations']) + 1,
                'image_id': img_id,
                'category_id': targets['labels'][j],
                'bbox': [x, y, w, h],
                'area': area,
                'iscrowd': 0  # Assuming all instances are not crowd
            })

    # Create a COCO object for your annotations
    coco_gt = COCO()
    coco_gt.dataset = coco_annotations
    coco_gt.createIndex()


    print('----------------------Model evaluation started--------------------------')

    device = next(model.parameters()).device
    model.eval()
    results = []

    with torch.no_grad():
        for images, targets in test_dataloader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            for i, output in enumerate(outputs):
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                image_id = targets[0]['image_id'].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    results.append({
                        'image_id': image_id,
                        'category_id': label,
                        'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        'score': score
                    })

    # Load your model's results into the COCOeval object
    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Get the evaluation metrics
    metrics = coco_eval.stats

    print('Evaluation metrics: AP = {:.4f}, AP50 = {:.4f}, AP75 = {:.4f}, APs = {:.4f}, APm = {:.4f}, APl = {:.4f}'.format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]))
    print('----------------------test ended--------------------------')






if __name__ == '__main__':
    main()
