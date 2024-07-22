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
import gc
gc.collect()
torch.cuda.empty_cache()

print("A")
print(gc.get_count())
for obj in gc.get_objects(generation=2):
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
    except:
        pass

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
print("B")
print(gc.get_count())
for obj in gc.get_objects(generation=2):
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
    except:
        pass

batch_size = 32
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
            train_rate_distortion(train_loader, test_loader, model_name, codebook_length, device, batch_size)
        elif function_run == 'show-rate-distortion':
            show_rate_distortion(test_loader, model_name, codebook_length, device, batch_size)
        else:
            print("Unknown Function")
    except RuntimeError:
        print("CUDA Out of Memory. Decreasing Batch Size by Half. New Batch Size: " + str(batch_size/2))
        batch_size = int(batch_size/2)
        del train_loader
        del test_loader
        gc.collect()
        print("C")
        print(gc.get_count())
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
            except:
                pass
            continue
        torch.cuda.empty_cache()
    break