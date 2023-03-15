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
    NumOfEpochs = 100
    BatchSize = 16
    num_workers = 0

    #SET Weights & Biases
    wandb.init(
        # set the wandb project where this run will be logged
        project="Master-Thesis",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
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
    validation_dataloader = DataLoader(validation_dataset, batch_size=BatchSize, shuffle=False, num_workers=num_workers,collate_fn=collate_fn)

    #load the model
    model = get_model_instance_segmentation(NumOfClasses)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

    train_loss = []
    val_loss = []


    print('----------------------train started--------------------------')

    for epoch in range(NumOfEpochs):
        start = time.time()
        model.train()
        model.to(device)
        i = 0    
        epoch_loss = 0
        for imgs, annotations in tqdm(training_dataloader):
            i += 1
            
            imgs =list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]


            loss_dict = model(imgs, annotations) 
            losses = sum(loss for loss in loss_dict.values())        

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            #lr_scheduler.step() 
            epoch_loss += losses
        
        train_loss.append(epoch_loss)

        # Validate the model
        #model#.eval()
        validation_loss = 0.0

        for imgs, annotations in tqdm(validation_dataloader):
            #imgs, annotations = imgs.to(device), annotations.to(device)
            i += 1
            imgs =list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

      

            with torch.no_grad():
                loss_dict_val = model(imgs, annotations)
                
                losses_val = sum(loss for loss in loss_dict_val.values())
                validation_loss += losses_val.item()
        
        val_loss.append(validation_loss)

        wandb.log({'epoch': epoch+1,"training_loss": epoch_loss,"validation_loss": validation_loss})

        print(f'Epoch {epoch+1}: train_loss={epoch_loss}, val_loss={validation_loss}, time : {time.time() - start}')
        # Save the lists to a pickle file
        # Create a dictionary containing the lists

        model.to("cpu")
        #save the model state
        torch.save(model.state_dict(),f'CNN_Model.pt')

        print("Model state saved on epoch: ", (epoch+1))


    print('----------------------train ended--------------------------')

    # Create a dictionary containing the lists
    data = {'val_loss': val_loss, 'epoch_loss': epoch_loss}

    # Save the lists to a pickle file
    with open('losses_CNN.pickle', 'wb') as f:
        pickle.dump(data, f)

    model.to("cpu")
    #save the model state
    torch.save(model.state_dict(),f'test.pt')





if __name__ == '__main__':
    main()
