# Hyperparameters for model training 
import torch

class Config(): 
    def __init__(self): 
        self.criterion = torch.nn.CrossEntropyLoss()