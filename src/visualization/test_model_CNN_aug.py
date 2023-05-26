import cv2
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
from src.models.utils import *
import numpy as np
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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

    score_threshold = 0.7
    iou_threshold = 0.5

    #load test data
    test_dataset = XRayDataSet(pathlib.Path('literature/Other/supervisely/wrist/test_pickles'))
    test_dataloader = DataLoader(test_dataset, batch_size=BatchSize, shuffle=False, num_workers=num_workers,collate_fn=collate_fn)



    #load the model state
    model = get_model_instance_segmentation(3)
    best_model = torch.load(f'Best_val_CNN_Model_aug.pt')
    model.load_state_dict(best_model['model_state_dict'])

    model.to(device)


    print('----------------------Model evaluation started--------------------------')


    print('Now we will evaluate the model based on the coco evaluation fucntion')


    #load the test coco dataset for the eval
    # Load the COCO object from a JSON file
    with open('test_coco_gt.json', 'r') as f:
        coco_gt_data = json.load(f)
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_data
    coco_gt.createIndex()

    print('Results with confidence = 0.7')
    validation_loss = testing_step(model,device,test_dataloader,coco_gt,0.7)



    print('----------------------Model evaluation ended--------------------------')




if __name__ == '__main__':
    main()
