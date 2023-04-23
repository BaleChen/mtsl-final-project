import torch
from torch import nn

class Model(nn.Module):
    # Basic version for now
    def __init__(self, model_name, num_class):
        super(Model,self).__init__()
        
        self.model = init_pretrained_model(model_name, num_class)
    
    def forward(self, x):
        return self.model(x)
    
def init_pretrained_model(model_name: str, num_class: int):
    """
    Loading pretrained model base on model name
    """
    torch.hub.set_dir("../.cache")
    
    if model_name == "resnet101":
        model = torch.hub.load("pytorch/vision:v0.13.1", "resnet101", weights="IMAGENET1K_V2")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_class)
        return model
    
    if model_name == "resnet50":
        model = torch.hub.load("pytorch/vision:v0.13.1", "resnet50", weights="IMAGENET1K_V2")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_class)
        return model
    
    elif model_name == "resnet18":
        model = torch.hub.load("pytorch/vision:v0.13.1", "resnet18", weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_class)
        return model
    
    elif model_name == "vgg16_bn":
        model = torch.hub.load("pytorch/vision:v0.13.1", "vgg16_bn", weights="IMAGENET1K_V1")
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_class)
        return model
    
    elif model_name == "vit_b_16":
        model = torch.hub.load("pytorch/vision:v0.13.1", "vit_b_16", weights="IMAGENET1K_V1")
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_class)
        return model
    
    else:
        raise Exception(f"Model {model_name} not supported yet.")
