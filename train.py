
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, datasets, models
import torchvision.models as models
import torch
import torch.nn as nn
from torch import optim 
import torch.utils.data as utils
import torch.utils.data as td
from torch.utils.data import DataLoader
from model import get_model 
from dataloader import get_dataset
import argparse


def save_checkpoint(model, image_datasets, optimizer, epochs ): 
    
    save_path = "/home/workspace/ImageClassifier/checkpoint.path"
    checkpoint = {
    "model_state_dict" : model.state_dict(),
    "class_to_idx" : image_datasets["train_data"].class_to_idx,
    "classifier" : model.classifier,
    "optimizer_state_dict" : optimizer.state_dict,
    "epochs" : epochs,
    "input_size" : 25088,
    "output_size" : 102 
    
    }
    torch.save(checkpoint, save_path)
  
   

def train(epochs = 3):   
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', action="store", default="./flowers/", type = str)
    
    image_path = parser.parse_args().image_path
    model = get_model()
    dataloaders = get_dataset(image_path)
    image_datasets = get_dataset(image_path, True)
    
    #Define Loss Function 
    criterion = nn.NLLLoss()

    #Define Optimizer 
    optimizer = optim.Adam(model.classifier.parameters(), lr = 4e-4)

    #Choose device: 
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device) 
    
    steps = 0
    print_every = 5
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in dataloaders["train"]: 
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0: 
                valid_loss = 0 
                accuracy = 0 
                model.eval()

                with torch.no_grad(): 
                    for inputs, labels in dataloaders["valid"]:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)

                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        #Accuracy 
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/len(dataloaders["train"])),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(dataloaders["valid"])),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders["valid"])))

                running_loss = 0
                model.train()
               
                #Save the checkpoint
                save_checkpoint(model, image_datasets, optimizer=optimizer, epochs=epoch)

                
                
if __name__ == "__main__":
    train()