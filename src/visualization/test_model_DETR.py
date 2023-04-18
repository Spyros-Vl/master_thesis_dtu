from cv2 import cv2
import os
import json
import time

import pathlib
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from master_thesis_dtu.src.data.my_rpg_dataset import CocoDetection
from master_thesis_dtu.src.data.my_rpg_dataset import collate_fn_COCO
from tqdm import tqdm

#for model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from transformers import DetrImageProcessor
from transformers import DetrConfig, DetrForObjectDetection
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
import wandb

import warnings
from src.models.utils import *
import numpy as np
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Mute all warnings
warnings.filterwarnings('ignore')

# filter out FutureWarning message from transformers
warnings.filterwarnings("ignore", message="The `max_size` parameter is deprecated and will be removed in v4.26.", category=FutureWarning)

def main():

    # train on the GPU or on the CPU, if a GPU is not available
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    train_dataset = CocoDetection(path_folder="data", processor=processor,status='test')
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_COCO, batch_size=1, shuffle=False,num_workers=4)

    from transformers import DetrConfig, DetrForObjectDetection

    config = DetrConfig.from_pretrained('facebook/detr-resnet-50',revision="no_timm",num_labels=2,id2label={0:"text",1:"fracture"},
                                                                ignore_mismatched_sizes=True) 
                                                                
    model = DetrForObjectDetection(config)
    best_model = torch.load(f'DETR_Model.pt',map_location=torch.device('cpu'))
    model.load_state_dict(best_model['model_state_dict'])
    print("The model best AP score with IoU = 0.5 in validation set was : ",best_model['best_loss'])

    model.to(device)


    print('----------------------Model evaluation started--------------------------')

    print('Dummy accuracy metric under development')

    print('Now we will evaluate the model based on the coco evaluation fucntion')

    evaluator = CocoEvaluator(coco_gt=train_dataset.coco, iou_types=["bbox"])

    print("Running evaluation...")
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # get the inputs
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # turn into a list of dictionaries (one item for each example in the batch)
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)
        # provide to metric
        # metric expects a list of dictionaries, each item 
        # containing image_id, category_id, bbox and score keys 
        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)
        evaluator.update(predictions)
        

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    coco_eval = evaluator.coco_eval['bbox']

    metrics = coco_eval.stats
    f = (coco_eval.eval['recall'][0][1][0][2]*100)
    t = (coco_eval.eval['recall'][0][0][0][2]*100)
    print("Secondary Evaluation Metrics :\n The AP value on IoU = 0.5 that matters the most is : {:.4f}.\n The Recall of the model for Text label is : {:.4f}. The Precision of the model for Text is : {:.4f}.\n The Recall of the model for Fracture label is : {:.4f}. The Precision of the model for Fracture is : {:.4f}".format(metrics[1], coco_eval.eval['recall'][0][0][0][2], coco_eval.eval['precision'][0][int(t)][0][0][2], coco_eval.eval['recall'][0][1][0][2], coco_eval.eval['precision'][0][int(f)][1][0][2] ))


    print("---Per-Class evaluation---")

    DETR_per_class(train_dataset,train_dataloader,device,model,processor)

    plot_curve(coco_eval)


    print('----------------------Model evaluation ended--------------------------')




if __name__ == '__main__':
    main()
