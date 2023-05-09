from cv2 import cv2
import os
import json
import time
import pickle

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
import wandb


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
    NumOfClasses = 3 
    NumOfEpochs = 50
    BatchSize = 32
    num_workers = 5

    #SET Weights & Biases
    wandb.init(
        # set the wandb project where this run will be logged
        project="Master-Thesis",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "Gradient_clip": 1,
        "architecture": "FastRCNN TorchVision",
        "dataset": "Pediatric wrist trauma X-ray",
        "epochs": NumOfEpochs,
        "BatchSize": BatchSize,
        "Optimizer": "SGD",
        }
    )
    
    
    #load train data
    train_dataset = XRayDataSet(pathlib.Path('literature/Other/supervisely/wrist/train_pickles'))
    training_dataloader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, num_workers=num_workers,collate_fn=collate_fn)

    #load validation data
    validation_dataset = XRayDataSet(pathlib.Path('literature/Other/supervisely/wrist/validation_pickles'))
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=4,collate_fn=collate_fn)

    #load the model
    model = get_model_instance_segmentation(NumOfClasses)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.AdamW(params, lr=0.005, betas=(0.9, 0.999), weight_decay=0.0005)

    optimizer = torch.optim.SGD(params, lr=0.01,momentum=0.9, weight_decay=0.0005)

    train_loss = []
    val_loss = []

    best_loss = 0

    #load the validation coco dataset for the eval
    # Load the COCO object from a JSON file
    with open('coco_gt.json', 'r') as f:
        coco_gt_data = json.load(f)
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_data
    coco_gt.createIndex()


    print('----------------------train started--------------------------')

    for epoch in range(NumOfEpochs):
        
        start = time.time()
        model.train()
        model.to(device)  
    
        #training step
        epoch_loss = train_one_epoch(model,training_dataloader,device,optimizer)
        train_loss.append(epoch_loss)

        #validation step
        
        validation_loss = validation_step(model,device,validation_dataloader,coco_gt)
        val_loss.append(validation_loss)

        wandb.log({'epoch': epoch+1,"training_loss": epoch_loss,"validation_loss": validation_loss})

        print(f'Epoch {epoch+1}: train_loss={epoch_loss}, val_loss={validation_loss}, time : {time.time() - start}')
        
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
            torch.save(checkpoint,f'Best_val_CNN_Model.pt')
            print("Model state saved on epoch: ", (epoch+1))



    print('----------------------train ended--------------------------')

    # save the model from last epoch to train more
    model.to("cpu")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }
    torch.save(checkpoint,f'Last_CNN_Model.pt')
    print("Model state saved on epoch: ", (epoch+1))


if __name__ == '__main__':
    main()
