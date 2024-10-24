import torch
from torchvision import datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import sys
from train import train
from test import test
from rate_distortion import train_rate_distortion
from rate_distortion import show_rate_distortion
import os
import matplotlib.pyplot as plt
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
codebook_length = 128
epochs = 20
function_run = sys.argv[1]
model_name = sys.argv[2]
if not(os.path.exists('models')):
    os.mkdir('models')
files = os.scandir('models')
for file in files:
    if (model_name + ".pth" == file.name):
        model_exist = True
        print("Model Found")
        break

batch_size = 16
curr_ind = 0
losses = []
while True:
    #Initialize Dataloader over training data
    train_loader = torch.utils.data.DataLoader(
        dataset = data,
        batch_size = batch_size, 
        sampler = train_sampler,
        num_workers = 4,
        pin_memory = True
    )

    #Initialize Dataloader over test data
    test_loader = torch.utils.data.DataLoader(
        dataset = data,
        batch_size = batch_size,
        sampler = test_sampler
    )
    try:    
        if function_run == 'train':
            is_show = False
            train(train_loader, model_name, codebook_length, device, model_exist, is_show, epochs, batch_size)
        elif function_run == 'testTrain':
            is_show = True
            epochs = 1
            train(train_loader, model_name, codebook_length, device, model_exist, is_show, epochs, batch_size)
        elif function_run == 'showtrain':
            is_show = True
            train(train_loader, model_name, codebook_length, device, model_exist, is_show, epochs, batch_size)
        elif function_run == 'test':
            is_show = True
            test(train_loader, model_name, codebook_length, device, is_show, batch_size)
        elif function_run == 'train-rate-distortion':
            codebook_vals = [2, 4, 8, 16, 32, 64,128]
            train_rate_distortion(train_loader, test_loader, model_name, codebook_vals[curr_ind], device, batch_size)
            if (curr_ind + 1 != len(codebook_vals)): 
                curr_ind += 1
                continue
        elif function_run == 'show-rate-distortion':
            codebook_vals = [2, 4, 8, 16, 32, 64, 128]
            loss = show_rate_distortion(test_loader, model_name, codebook_vals[curr_ind], device, batch_size)
            losses.append(loss)
            if (curr_ind + 1 != len(codebook_vals)): 
                curr_ind += 1
                continue
            else:
                plt.plot(codebook_vals, losses)
                plt.xlabel('Codebook Values')
                plt.ylabel('Loss')
                plt.title('Rate-Distortion')
                plt.show()
        else:
            print("Unknown Function")
    except RuntimeError:
        print("CUDA Out of Memory. Decreasing Batch Size by Half. New Batch Size: " + str(batch_size/2))
        batch_size = int(batch_size/2)
        continue
    break