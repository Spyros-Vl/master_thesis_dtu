import numpy as np
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
from src.data.my_rpg_dataset import XRayDataSet
from src.data.my_rpg_dataset import collate_fn
from tqdm import tqdm

#for model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.faster_rcnn import *
import torch

import contextlib
import io

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from coco_eval import *



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

def create_COCO_from_dataset(img_list,ann_list,status):
    
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
            "id": 0,
            "name": "text"
        },
        {
            "id": 1,
            "name": "fracture"
        }]
    
    anno_id = 1

    images = []
    annotations = []
    
    for idx in tqdm(range(len(img_list))):

        img_path = os.path.join(img_list[idx])
        ann_path = os.path.join(ann_list[idx])

        #fix the images format

        image_id = idx + 1
        file_name = img_path

        # Load the image using cv2
        image = cv2.imread(img_path)

        # Get the height and width of the image
        height, width, channels = image.shape


        img = {
                    "id": image_id,
                    "file_name": file_name.replace("\\","/"),
                    "width": width,
                    "height": height
                }
        
        images.append(img)
            
        classes = {'text' : 0, 'fracture' : 1}


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
    with open((status + '_coco_data.json'), 'w') as f:
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

def train_one_epoch(model,training_dataloader,device,optimizer):
    epoch_loss = 0
    for imgs, annotations in tqdm(training_dataloader):
            
        imgs =list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]


        loss_dict = model(imgs, annotations) 
        losses = sum(loss for loss in loss_dict.values())        

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses
        
    return epoch_loss

def validation_step(model,device,validation_dataloader,coco_gt):

    model.eval()
    results = [] 
    with torch.no_grad():
        for images, targets in tqdm(validation_dataloader):
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
                        'image_id': image_id[0],
                        'category_id': label,
                        'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        'score': score
                    })

    # Load your model's results into the COCOeval object
    coco_dt = coco_gt.loadRes(results)

    # Create a COCOeval object for computing mAP
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')


    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Get the evaluation metrics
    metrics = coco_eval.stats

    print('Evaluation metrics: AP = {:.4f}, AP50 = {:.4f}, AP75 = {:.4f}, APs = {:.4f}, APm = {:.4f}, APl = {:.4f}'.format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]))

    return metrics[1]

def testing_step(model,device,validation_dataloader,coco_gt,confidence):
    
    model.eval()
    results = [] 
    with torch.no_grad():
        for images, targets in tqdm(validation_dataloader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            for i, output in enumerate(outputs):
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                image_id = targets[0]['image_id'].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    if score > confidence :
                        results.append({
                            'image_id': image_id[0],
                            'category_id': label,
                            'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                            'score': score
                        })

    
    # Load your model's results into the COCOeval object
    coco_dt = coco_gt.loadRes(results)

    # Create a COCOeval object for computing mAP
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    print('---Normal Eval---')
    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Get the evaluation metrics
    metrics = coco_eval.stats

    print('Evaluation metrics: AP = {:.4f}, AP50 = {:.4f}, AP75 = {:.4f}, APs = {:.4f}, APm = {:.4f}, APl = {:.4f}'.format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]))

    plot_curve(coco_eval,"cnn_plot")
    
    #print the recall and the precision
    f = (coco_eval.eval['recall'][0][0][0][2]*100)
    t = (coco_eval.eval['recall'][0][1][0][2]*100)
    print("Secondary Evaluation Metrics :\n The AP value on IoU = 0.5 that matters the most is : {:.4f}.\n The Recall of the model for Text label is : {:.4f}. The Precision of the model for Text is : {:.4f}.\n The Recall of the model for Fracture label is : {:.4f}. The Precision of the model for Fracture is : {:.4f}".format(metrics[1], coco_eval.eval['recall'][0][1][0][2], coco_eval.eval['precision'][0][int(t)][1][0][2], coco_eval.eval['recall'][0][0][0][2], coco_eval.eval['precision'][0][int(f)][0][0][2] ))

    id2label = id2label={2:"text",1:"fracture"}
    print('---Per-Class Eval---')
    for catId in coco_gt.getCatIds():
        
        coco_eval.params.catIds = [catId]
        
        # Redirect standard output to a null device
        with contextlib.redirect_stdout(io.StringIO()):
            # Call the summarize() function
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        # Get the evaluation metrics
        metrics = coco_eval.stats
        print("The AP value on IoU = 0.5 for class {} is : {:.4f}.\n".format(id2label[catId],metrics[1]))

    


    return metrics[1]

def train_one_epoch_DETR(model,train_dataloader,device,optimizer):

    epoch_loss = 0

    for idx, batch in enumerate(tqdm(train_dataloader)):
            
        # get the inputs
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        epoch_loss += loss
    
    return epoch_loss

def validation_step_DETR(model,device,validation_dataset,validation_dataloader,processor):

    evaluator = CocoEvaluator(coco_gt=validation_dataset.coco, iou_types=["bbox"])

    print("Running evaluation...")
    for idx, batch in enumerate(tqdm(validation_dataloader)):
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

        if results[0]['boxes'].numel() == 0:
            print("Validation skipped")
            return 0
        # provide to metric
        # metric expects a list of dictionaries, each item 
        # containing image_id, category_id, bbox and score keys 
        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)
        evaluator.update(predictions)

   
    evaluator.synchronize_between_processes()
    # Run evaluation
    evaluator.accumulate()
    evaluator.summarize()

    evaluator.coco_eval['bbox'].evaluate()
    metrics =evaluator.coco_eval['bbox'].stats


    print('Evaluation metrics: AP = {:.4f}, AP50 = {:.4f}, AP75 = {:.4f}, APs = {:.4f}, APm = {:.4f}, APl = {:.4f}'.format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]))

    return metrics[1]


def plot_curve(coco_eval,filename):
    all_precision = coco_eval.eval['precision']

    pr_5 = all_precision[0, :, 0, 0, 2] # data for IoU@0.5
    pr_7 = all_precision[4, :, 0, 0, 2] # data for IoU@0.7
    pr_9 = all_precision[8, :, 0, 0, 2] # data for IoU@0.9

    x = np.arange(0, 1.01, 0.01)

    plt.figure(figsize=(16,6))

    plt.subplot(1, 2, 1)
    plt.plot(x, pr_5, label='IoU@0.5')
    plt.plot(x, pr_7, '--g',label='IoU@0.7')
    plt.plot(x, pr_9, '--r',label='IoU@0.9')
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.plot([0,1],  [1,1], linestyle='dashed', dashes=(1,3),color='black')
    plt.plot([1,1],  [0,1], linestyle='dashed', dashes=(1,3),color='black')
    plt.scatter(coco_eval.eval['recall'][0][0][0][2], coco_eval.eval['precision'][0][int(coco_eval.eval['recall'][0][0][0][2]*100)][0][0][2], label = "Best Trade-off",color = "blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR-Curve for Fracture label")

    pr_5 = all_precision[0, :, 1, 0, 2] # data for IoU@0.5
    pr_7 = all_precision[4, :, 1, 0, 2] # data for IoU@0.7
    pr_9 = all_precision[8, :, 1, 0, 2] # data for IoU@0.9

    plt.subplot(1, 2, 2)
    plt.plot(x, pr_5, label='IoU@0.5')
    plt.plot(x, pr_7, '--g',label='IoU@0.7')
    plt.plot(x, pr_9, '--r',label='IoU@0.9')
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.plot([0,1],  [1,1], linestyle='dashed', dashes=(1,3),color='black')
    plt.plot([1,1],  [0,1], linestyle='dashed', dashes=(1,3),color='black')
    plt.scatter(coco_eval.eval['recall'][0][1][0][2], coco_eval.eval['precision'][0][int(coco_eval.eval['recall'][0][1][0][2]*100)][1][0][2], label = "Best Trade-off",color = "blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR-Curve for Text label")

    plt.suptitle("PR-Curve")
    plt.savefig(f'{filename}.png')



def DETR_per_class(train_dataset,train_dataloader,device,model,processor):
    id2label={0:"text",1:"fracture"}
    print('---Per-Class Eval---')
    for catId in train_dataset.coco.getCatIds():
    
        # Redirect standard output to a null device
        with contextlib.redirect_stdout(io.StringIO()):
            # Call the summarize() function
            evaluator = CocoEvaluator(coco_gt=train_dataset.coco, iou_types=["bbox"])
            evaluator.coco_eval['bbox'].params.catIds = [catId]

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
        # Get the evaluation metrics
        coco_eval = evaluator.coco_eval['bbox']
        metrics = coco_eval.stats
        print("The AP value on IoU = 0.5 for class {} is : {:.4f}.\n".format(id2label[catId],metrics[1]))

def get_model_instance_segmentation(num_classes):
      
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model