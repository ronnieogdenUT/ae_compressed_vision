import torch
import torch.nn as nn
#from pytorch_msssim import ms_ssim
from autoencoder import Autoencoder
from show import show
from torchvision import datasets
import time
from datetime import timedelta
import os

#Import MovingMNIST Dataset
data = datasets.MovingMNIST(
    root = "./data", 
    download = True
)

video = torch.squeeze(data[0], dim=0) # 20 frames
batch_size = 2
print("Video Shape: " + str(video.shape))

in_channels = 1
codebook_length = 128
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
model_name = "model128.pth"
model_path = os.path.join('models', model_name)
model = Autoencoder(in_channels, codebook_length, device, batch_size).to(device) #Intialize Model
model.load_state_dict(torch.load(model_path))
model.eval()
i=0
while True:
    #GET 2 FRAMES(2 x 1 x 64 x 64)
    frameSet = video[i:i+1]
    print(frameSet.shape)
    frameSet.to(device)
    start = time.perf_counter()
    reconstructed = model(frameSet)
    end = time.perf_counter()
    print("Time Elapsed: " + str(timedelta(seconds = end-start)))
    i += 2