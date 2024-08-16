import torch
import torch.nn as nn
#from pytorch_msssim import ms_ssim
from autoencoder import Autoencoder
from show import show
from torchvision import datasets
import time
from datetime import timedelta
import os
import torchvision

#Import Dataset
data = torch.load("pytorch.pt")
# 2694 x 3 x 260 x 346
batch_size = 5
train_loader = torch.utils.data.DataLoader(
        dataset = data,
        batch_size = batch_size, 
        num_workers = 4,
        pin_memory = True
    )

in_channels = 1
codebook_length = 4
#Check CUDA Availability
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#Create and IMPORT MODEL
model_name = "model4.pth"
model_path = os.path.join('models', model_name)
model = Autoencoder(in_channels, codebook_length, device, batch_size).to(device) #Intialize Model
model.load_state_dict(torch.load(model_path))
model.eval()
i=0
for video in train_loader:
    while True:
        #Input 2 Frames x L x W x C
        #Permute to 2 x C x L x W
        frameSet = video.to(device)
        start = time.perf_counter()
        frameSet = frameSet.to(torch.float32)
        frameSet = torch.permute(frameSet, (0, 3, 1, 2))
        frameSet = torchvision.transforms.functional.rgb_to_grayscale(frameSet, num_output_channels = 1)
        frameSet = torch.permute(frameSet, (1, 0, 2, 3))
        reconstructed = model(frameSet)
        end = time.perf_counter()
        print("Time Elapsed: " + str(timedelta(seconds = end-start)))
        i += 2