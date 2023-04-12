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
    num_workers =5

    score_threshold = 0.8
    iou_threshold = 0.5

    #load test data
    test_dataset = XRayDataSet(pathlib.Path('literature/Other/supervisely/wrist/test_pickles'))
    test_dataloader = DataLoader(test_dataset, batch_size=BatchSize, shuffle=False, num_workers=num_workers,collate_fn=collate_fn)



    #load the model state
    model = get_model_instance_segmentation(3)
    model.load_state_dict(torch.load(f'CNN_Model.pt'))
    model.to(device)


    print('----------------------Model evaluation started--------------------------')

    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(test_dataloader):
        
            images =list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
            outputs = model(images)

            for i, output in enumerate(outputs):
                boxes = output['boxes'].cpu()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                target_boxes = targets[i]['boxes'].cpu()
                target_labels = targets[i]['labels'].cpu().numpy()
                total += target_labels.size
                for box, score, label in zip(boxes, scores, labels):
                    if label in target_labels:
                        index = np.where(target_labels == label)[0][0]
                        if score > score_threshold and box_iou(box.unsqueeze(0), target_boxes[index].unsqueeze(0)) > iou_threshold:
                            correct += 1

    accuracy = 100 * correct / total

    print('----------------------train ended--------------------------')
    print('The total model accuracy in the test set was: ', accuracy)






if __name__ == '__main__':
    main()
