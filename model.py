import torchvision
from torchvision import models
import torchvision.models as models
import torch
import torch.nn as nn

def get_model():
    #Set device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the pre-trained model
    model = torchvision.models.vgg16(pretrained=True)

    #Freeze the pre-traied layers 
    for parameter in model.parameters(): 
        parameter.requires_grad = False

    #Define a Classifier for the model 
    Classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(0.5), 
        nn.Linear(1024, 102),
        nn.LogSoftmax(dim=1)
    )

    #Replace the pre-trained model's classifier 
    model.classifier = Classifier 
    model.to(device)
    
    
    return model 
