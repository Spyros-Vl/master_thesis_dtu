from cv2 import cv2
import os
import json
import time
import pickle

import pathlib
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from master_thesis_dtu.src.data.my_dataset import XRayDataSet
from master_thesis_dtu.src.data.my_dataset import collate_fn
from tqdm import tqdm

#for model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
import torch
from utils import *


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
    NumOfEpochs = 1
    BatchSize = 1
    num_workers = 1

    #load train data
    test = XRayDataSet(pathlib.Path('literature/Other/supervisely/wrist/train_pickles'))
    train_dataset = XRayDataSet(pathlib.Path('literature/Other/supervisely/wrist/train_pickles'))
    training_dataloader = DataLoader(test, batch_size=BatchSize, shuffle=True, num_workers=num_workers,collate_fn=collate_fn)

    #load validation data
    validation_dataset = XRayDataSet(pathlib.Path('literature/Other/supervisely/wrist/validation_pickles'))
    validation_dataloader = DataLoader(test, batch_size=BatchSize, shuffle=False, num_workers=num_workers,collate_fn=collate_fn)

    #load test data
    test_dataset = XRayDataSet(pathlib.Path('literature/Other/supervisely/wrist/test_pickles'))
    test_dataloader = DataLoader(test, batch_size=BatchSize, shuffle=False, num_workers=num_workers,collate_fn=collate_fn)



    #load the model
    model = get_model_instance_segmentation(NumOfClasses)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

    train_loss = []
    val_loss = []


    print('----------------------train started--------------------------')

    for epoch in range(NumOfEpochs):
        start = time.time()
        model.train()
        i = 0    
        epoch_loss = 0
        for imgs, annotations in tqdm(training_dataloader):
            #imgs, annotations = imgs.to(device), annotations.to(device)
            i += 1
            imgs =list(img.squeeze(dim=0).to(device) for img in imgs)
            annotations = [{k: v for k, v in t[0].items()} for t in annotations]

            ####-----------MOVE annotations to device---------------#####

            # Iterate over the list of dicts and move each tensor to the device
            for annotation in annotations:
                for key, value in annotation.items():
                    if isinstance(value, torch.Tensor):
                        annotation[key] = value.to(device)
                
            ####-----------MOVE annotations to device---------------#####

            loss_dict = model(imgs, annotations) 
            print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())        

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            lr_scheduler.step() 
            epoch_loss += losses
        
        train_loss.append(epoch_loss)

        # Validate the model
        model#.eval()
        validation_loss = 0.0

        for imgs, annotations in tqdm(validation_dataloader):
            #imgs, annotations = imgs.to(device), annotations.to(device)
            i += 1
            imgs =list(img.squeeze(dim=0).to(device) for img in imgs)
            annotations = [{k: v for k, v in t[0].items()} for t in annotations]

            ####-----------MOVE annotations to device---------------#####

            # Iterate over the list of dicts and move each tensor to the device
            for annotation in annotations:
                for key, value in annotation.items():
                    if isinstance(value, torch.Tensor):
                        annotation[key] = value.to(device)

            with torch.no_grad():
                loss_dict_val = model(imgs, annotations)
                print(loss_dict_val)
                losses_val = sum(loss for loss in loss_dict_val.values())
                validation_loss += losses_val.item()
        
        val_loss.append(validation_loss)

        print(f'Epoch {epoch+1}: train_loss={epoch_loss}, val_loss={validation_loss}, time : {time.time() - start}')


    print('----------------------train ended--------------------------')

    # Save the lists to a pickle file
    with open('losses.pickle', 'wb') as f:
        pickle.dump((train_loss, val_loss), f)

    #save the model state
    torch.save(model.state_dict(),f'test.pt')





if __name__ == '__main__':
    main()