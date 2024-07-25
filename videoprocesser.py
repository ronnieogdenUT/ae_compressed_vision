import torch
import torch.nn as nn
#from pytorch_msssim import ms_ssim
from autoencoder import Autoencoder
from show import show
from torchvision import datasets
import time
from datetime import timedelta

#Import MovingMNIST Dataset
data = datasets.MovingMNIST(
    root = "./data", 
    download = True
)

video = data[0] # 20 frames
batch_size = 2

train_loader = torch.utils.data.DataLoader(
        dataset = data,
        batch_size = batch_size, 
        num_workers = 4,
        pin_memory = True
    )

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
model_path = "model128.pth"
model = Autoencoder(in_channels, codebook_length, device, batch_size).to(device) #Intialize Model
model.load_state_dict(torch.load(model_path))
model.eval()

while True:
    try:
        #GET 2 FRAMES(2 x 1 x 64 x 64)
        frameSet = video[0,1]
        frameSet.to(device)
        start = time.perf_counter()
        reconstructed = model(frameSet)
        end = time.perf_counter()
        print("Time Elapsed: " + str(timedelta(seconds = end-start)))
    except:
        break