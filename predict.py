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
import random 
import os
from PIL import Image
from utils import parser
from model import get_model 
from dataloader import get_dataset
import argparse
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(file_path): 
    #Load the checkpoint 
    checkpoint = torch.load(file_path)
    #Rebuild the model
    model = models.vgg16(pretrained=True)
    
    #Freeze pre-trained layers
    for parameter in model.parameters(): 
        parameter.requires_grad = False
        
    model.classifier = checkpoint["classifier"]
    model.load_state_dict = checkpoint["model_state_dict"]
    model.class_to_idx = checkpoint["class_to_idx"]
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    #normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    
    
    #Open the image
    pil_image = Image.open(image)
    
    #Preprocess the image
    pil_image = preprocess(pil_image)
    #pil_image = normalize(pil_image)
    
    return pil_image.numpy()


def predict(image_path, checkpoint, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
   
    model = load_checkpoint(checkpoint) 
    model.to(device)
    model.eval() 
 
    #test_dir = os.path.join(data_dir, 'test')
    
    #Process the image and convert to tensor 
    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float()
    image = image.to(device)
    
    #get log probabilities
    with torch.no_grad(): 
        logps = model.forward(image)
        
    
    #Calculate probabilities using x.topk
    probability = torch.exp(logps).cpu().data
    #print(probability)
    
    topk_probabilities, topk_classes  = probability.topk(topk)
    
    print(topk_probabilities, topk_classes)
    
    topk_probabilities = topk_probabilities.cpu().numpy()
    topk_classes = topk_classes.cpu().numpy()
    
    return topk_probabilities, topk_classes 

def classify_image():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', action="store", default="./flowers/", type = str)
    parser.add_argument('checkpoint', default='./checkpoint.pth', action="store", type = str)   
    
    image_path = parser.parse_args().image_path
    checkpoint = parser.parse_args().checkpoint
    top_k = 5
    
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    path_elements = image_path.split('/')
    # Extract the label index (assuming it's always the third element)
    label_index = path_elements[6]
  
    flower_name = cat_to_name[label_index]
  
    
    
    #Label mapping
    #with open(args.category_names, 'r') as json_file:
     #   cat_to_name = json.load(json_file)
        
    topk_probabilities, topk_classes = predict(image_path, checkpoint)
    labels = []
    for class_index in topk_classes[0]: 
        #print(class_index)
        labels.append(cat_to_name[str(class_index)])
        
    
    print("Actual flower: {}".format(flower_name))
    print("Predicted flowers:")
    for k in range(top_k):
        print("{} with a probability of {:.2e}".format(labels[k], topk_probabilities[0][k]))
    
classify_image()

#SAMPLE PATHS
#image_path = "/home/workspace/ImageClassifier/flowers/test/102/image_08004.jpg"
#checkpoint_path = "/home/workspace/ImageClassifier/checkpoint.path"
#image_path = "/home/workspace/ImageClassifier/flowers/test/7/image_07211.jpg"
