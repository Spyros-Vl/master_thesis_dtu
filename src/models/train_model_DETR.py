from cv2 import cv2
import os
import json
import time
import pickle

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
from utils import *

import warnings

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




    #defines
    NumOfClasses = 2 
    NumOfEpochs = 10
    BatchSize = 1
    num_workers = 0
    checkpoint = "facebook/detr-resnet-50"

    processor = DetrImageProcessor.from_pretrained(checkpoint)

    train_dataset = CocoDetection(path_folder="data", processor=processor,train=True)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_COCO, batch_size=BatchSize, shuffle=True,num_workers=num_workers)



    #load the model
    model = DetrForObjectDetection.from_pretrained(checkpoint,num_labels=NumOfClasses,ignore_mismatched_sizes=True)    
    model.to(device)

    #params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
    param_dicts = [{"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},{"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],"lr": lr_backbone=1e-5}]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
    # and a learning rate scheduler
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

    train_loss = []
    val_loss = []


    print('----------------------train started--------------------------')

    for epoch in range(NumOfEpochs):
        start = time.time()
        model.train()
        i = 0    
        epoch_loss = 0
        for idx, batch in enumerate(tqdm(train_dataloader)):
            
            # get the inputs
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            t = labels[0]['boxes']
    
            # Check if the tensor contains NaNs
            has_nans = torch.isnan(t).any().item()

            if has_nans:
                print("The tensor contains NaNs.")
                print(t)
                continue

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_scheduler.step() 
            epoch_loss += loss

            
        
        train_loss.append(epoch_loss)

        print(f'Epoch {epoch+1}: train_loss={epoch_loss}, time : {time.time() - start}')

        model.to("cpu")
        #save the model state
        torch.save(model.state_dict(),f'DETR_Model.pt')

        print("Model state saved on epoch: ", (epoch+1))


    print('----------------------train ended--------------------------')

    val_loss = 0

    # Create a dictionary containing the lists
    data = {'val_loss': val_loss, 'epoch_loss': epoch_loss}

    # Save the lists to a pickle file
    with open('losses_DETR.pickle', 'wb') as f:
        pickle.dump(data, f)

    model.to("cpu")
    #save the model state
    torch.save(model.state_dict(),f'DETR_Model.pt')





if __name__ == '__main__':
    main()
