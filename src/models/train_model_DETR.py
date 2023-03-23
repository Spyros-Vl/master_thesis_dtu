from cv2 import cv2
import os
import json
import time
import pickle

import pathlib
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from src.data.my_rpg_dataset import CocoDetection
from src.data.my_rpg_dataset import collate_fn_COCO
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
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
import wandb

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
    NumOfEpochs = 100
    BatchSize = 16
    num_workers = 8
    checkpoint = "facebook/detr-resnet-50"

    #SET Weights & Biases
    wandb.init(
        # set the wandb project where this run will be logged
        project="Master-Thesis",
        
        # track hyperparameters and run metadata
        config={
        "architecture": "DETR model from Transformers",
        "dataset": "Pediatric wrist trauma X-ray",
        "epochs": NumOfEpochs,
        "BatchSize": BatchSize,
        "Optimizer": "ADAM",
        }
    )

    processor = DetrImageProcessor.from_pretrained(checkpoint)

    train_dataset = CocoDetection(path_folder="data", processor=processor,train=True)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_COCO, batch_size=BatchSize, shuffle=True,num_workers=num_workers)

    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}

    #load the model
    model = DetrForObjectDetection.from_pretrained(checkpoint,revision="no_timm",num_labels=len(id2label),id2label={0:"text",1:"fracture"},
                                                             ignore_mismatched_sizes=True)    
    model.to(device)

   
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                 "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                 "lr": 1e-5,
            },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4,weight_decay=1e-4)
   

    train_loss = []
    val_loss = []

    # watch the model and optimizer
    wandb.watch(model, log="all")

    print('----------------------train started--------------------------')

    for epoch in range(NumOfEpochs):
        start = time.time()
        model.train()
        model.to(device)
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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update learning rate
            #lr_scheduler.step() 
            epoch_loss += loss

            
        wandb.log({'epoch': epoch+1,"training_loss": epoch_loss})

        train_loss.append(epoch_loss)

        print(f'Epoch {epoch+1}: train_loss={epoch_loss}, time : {time.time() - start}')

        model.to("cpu")
        #save the model state
        torch.save(model.state_dict(),f'DETR_Model.pt')

        print("Model state saved on epoch: ", (epoch+1))

    wandb.finish()
    
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
