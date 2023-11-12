#Define data directories
import torchvision
from torchvision import transforms, datasets, models
from torch import optim 
import torch.utils.data as utils
import torch.utils.data as td
from torch.utils.data import DataLoader
from PIL import Image
import torch

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def get_dataset(data_dir, image_datasets_wanted=False):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets   
    training_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(), 
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             ])

    validation_test_transforms = transforms.Compose([transforms.Resize(256), #Resize to 256 
                                                transforms.CenterCrop(224), #Crop the centre to 224x224 
                                                transforms.ToTensor(), 
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])




    # TODO: Load the datasets with ImageFolder
    
    image_datasets = {"train_data" : datasets.ImageFolder(train_dir, transform = training_transforms),
                      "valid_data" : datasets.ImageFolder(valid_dir, transform = validation_test_transforms), 
                      "test_data" : datasets.ImageFolder(test_dir, transform = validation_test_transforms)
                     }
    if image_datasets_wanted:
        return image_datasets

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    batch_size = 64 
    dataloaders = {"train" : DataLoader(image_datasets["train_data"], batch_size = batch_size, shuffle = True),
                  "valid" : DataLoader(image_datasets["valid_data"], batch_size = batch_size, shuffle = True), 
                  "test" : DataLoader(image_datasets["test_data"], batch_size = batch_size, shuffle = True) 
                  }

    return dataloaders

def get_cat(cat_path):
    import json
    with open(cat_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
                        
