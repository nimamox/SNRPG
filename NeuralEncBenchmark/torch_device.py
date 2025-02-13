import os
import torch
import torch.nn as nn
import torchvision as tv

dtype = torch.float

device = os.getenv('device', 'cpu')
print('*, device')

if device == 'gpu':
    if torch.cuda.is_available():
        device = torch.device("cuda")     
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
    
print('NENC device:', device)
