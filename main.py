import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
import torch.nn.functional as f
import numpy as np
import torch.nn as nn
import matplotlib.animation as animation
import math
#from pytorch_msssim import ms_ssim
from torch.utils.data.sampler import SubsetRandomSampler
import gc
import sys
from train import train
from test import test
from rate_distortion import rate_distortion
from rate_distortion import show_rate_distortion
import os
torch.cuda.empty_cache()

#Import MovingMNIST Dataset
data = datasets.MovingMNIST(
    root = "./data", 
    download = True
)

#Split into Training and Test Datasets
test_split = 0.2
dataset_size = len(data)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))

#Shuffle Dataset
np.random.seed()
np.random.shuffle(indices)

train_indices, test_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

#Initialize Dataloader over training data
batch_size = 16
train_loader = torch.utils.data.DataLoader(
    dataset = data,
    batch_size = batch_size, 
    sampler = train_sampler
)

#Initialize Dataloader over test data
test_loader = torch.utils.data.DataLoader(
    dataset = data,
    batch_size = batch_size,
    sampler = test_sampler
)

#Check CUDA Availability
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#Call Main Function
model_exist = False
codebook_length = 20
epochs = 10
function_run = sys.argv[1]
model_name = sys.argv[2]
if not(os.path.exists('models')):
    os.mkdir('models')
files = os.scandir('models')
for file in files:
    if (model_name in file.name):
        model_exist = True
        print("Model Found")
        break

    
if function_run == 'train':
    is_show = False
    train(train_loader, model_name, codebook_length, device, model_exist, is_show, epochs)
elif function_run == 'showtrain':
    is_show = True
    train(train_loader, model_name, codebook_length, device, model_exist, is_show, epochs)
elif function_run == 'test':
    test(train_loader, model_name, codebook_length, device)
elif function_run == 'train-rate-distortion':
    rate_distortion(train_loader, test_loader, model_name, codebook_length, device)
elif function_run == 'show-rate-distortion':
    show_rate_distortion(test_loader, model_name, codebook_length, device)
else:
    print("Unknown Function")