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
    NumOfEpochs = 60
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
        "Optimizer": "ADAMW",
        }
    )

    processor = DetrImageProcessor.from_pretrained(checkpoint)

    train_dataset = CocoDetection(path_folder="data", processor=processor,status="train")
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_COCO, batch_size=BatchSize, shuffle=True,num_workers=num_workers)

    validation_dataset = CocoDetection(path_folder="data", processor=processor,status="validation")
    validation_dataloader = DataLoader(validation_dataset, collate_fn=collate_fn_COCO, batch_size=1, shuffle=False,num_workers=num_workers)

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
    best_loss = 0
    validation_loss = 0

    # watch the model and optimizer
    wandb.watch(model, log="all")

    print('----------------------train started--------------------------')

    for epoch in range(NumOfEpochs):
        start = time.time()
        model.train()
        model.to(device)   
        epoch_loss = 0

        #train one epoch
        epoch_loss = train_one_epoch_DETR(model,train_dataloader,device,optimizer)

        #validate the model
        #validation step
        if epoch > 3:
            validation_loss = validation_step_DETR(model,device,validation_dataset,validation_dataloader,processor)
            val_loss.append(validation_loss)
            
        wandb.log({'epoch': epoch+1,"training_loss": epoch_loss,"validation_loss": val_loss})

        train_loss.append(epoch_loss)

        print(f'Epoch {epoch+1}: train_loss={epoch_loss}," validation_loss=": {validation_loss}, time : {time.time() - start}')

        # check if the current validation loss is the new best
        if validation_loss > best_loss:
            best_loss = validation_loss
            # save the model checkpoint
            model.to("cpu")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }
            torch.save(checkpoint,f'Best_val_DETR_Model.pt')
            print("Model state saved on epoch: ", (epoch+1))

    wandb.finish()
    
    print('----------------------train ended--------------------------')

    # save the model from last epoch to train more
    model.to("cpu")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }
    torch.save(checkpoint,f'Last_DETR_Model.pt')
    print("Model state saved on epoch: ", (epoch+1))



if __name__ == '__main__':
    main()
