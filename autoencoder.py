import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
import torch.nn.functional as f
import numpy as np
import torch.nn as nn
import matplotlib.animation as animation
import math


train_data = datasets.MovingMNIST(
    root = "./data", 
    download = True
)

# test_data = datasets.MovingMNIST(
#     root = "./data", 
#     split = "test", 
#     download = True
# )

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size = batch_size, 
    shuffle = True
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class resblock_a(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_stack = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, 3, stride=1, padding='same'),
			torch.nn.BatchNorm3d(128),
			torch.nn.ReLU(),
            torch.nn.Conv3d(128, 128, 3, stride=1, padding='same'),
			torch.nn.BatchNorm3d(128)
        )
        
    def forward(self, x):
        residual = x
        out = self.conv_stack(x) + residual
        return out
    
class resblock_b(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resblock_a_stack = torch.nn.Sequential(
            resblock_a(),
            resblock_a(),
            resblock_a()
        )
        
    def forward(self, x):
        residual = x
        out = self.resblock_a_stack(x) + residual
        return out
    
class resblock_c(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resblock_stack = torch.nn.Sequential(
            resblock_b(),
            resblock_b(),
            resblock_b(),
            resblock_b(),
            resblock_b(),
            resblock_a()
        )
        
    def forward(self, x):
        residual = x
        out = self.resblock_stack(x) + residual
        return out

class Autoencoder(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        #Encoder
        self.encoderConv1 = torch.nn.Conv3d(in_channels, 64, 5, stride=(1,2,2))
        self.encoderBn1 = torch.nn.BatchNorm3d(64)
        self.encoderConv2 = torch.nn.Conv3d(64, 128, 5, stride=(1,2,2))
        self.encoderBn2 = torch.nn.BatchNorm3d(128)
        self.encoderConv3 = torch.nn.Conv3d(128, 32, 5, stride=(1,2,2))
        self.encoderBn3 = torch.nn.BatchNorm3d(32)   
        
        # Figure out if resblock_c needs to be a transposed version... I think they are the same here		
        #Decoder
        self.decoderConv1 = torch.nn.ConvTranspose3d(32, 128, 3, stride=(1,2,2), padding=1, output_padding=(0,1,1))
        self.decoderBn1 = torch.nn.BatchNorm3d(128)
        self.decoderConv2 = torch.nn.ConvTranspose3d(128, 64, 5, stride=(1,2,2), padding=2, output_padding=(0,1,1))
        self.decoderBn2 = torch.nn.BatchNorm3d(64)
        self.decoderConv3 = torch.nn.ConvTranspose3d(64, in_channels, 5, stride=(1,2,2), padding=2, output_padding=(0,1,1))
        self.decoderBn3 = torch.nn.BatchNorm3d(in_channels)

    def forward(self, x):
        stride = (1,2,2)

        #Encoder
        x = f.pad(x, self.same_pad(x, stride, 5))
        x = self.encoderConv1(x)
        x = self.encoderBn1(x)
        x = f.relu(x)
        
        x = f.pad(x, self.same_pad(x, stride, 5))
        x = self.encoderConv2(x)
        x = self.encoderBn2(x)
        x = f.relu(x)
        resblock_c()

        x = f.pad(x, self.same_pad(x, stride, 5))
        x = self.encoderConv3(x)
        x = self.encoderBn3(x)

        #Decoder
        x = self.decoderConv1(x)
        x = self.decoderBn1(x)
        x = f.relu(x)
        resblock_c()

        x = self.decoderConv2(x)
        x = self.decoderBn2(x)
        x = f.relu(x)

        x = self.decoderConv3(x)
        x = self.decoderBn3(x)
        
        return x
    
    #Calculates Padding(Mimics Tensor Flow padding = 'same')
    def same_pad(self, x, stride, kernel):
        size = x.size()
        in_height = size[4]
        in_width = size[3]
        filter_height = kernel
        filter_width = kernel

        out_height = math.ceil(float(in_height) / float(stride[1]))
        out_width  = math.ceil(float(in_width) / float(stride[2]))

        #3D Padding: Time Dimension, Front, Back
        pad_forward = (kernel-1)//2
        pad_backward = (kernel-1)//2

        
        #Regular 2D Padding: Top, Bottom, Left, Right
        pad_along_height = max((out_height - 1) * stride[1] + filter_height - in_height, 0)
        pad_along_width = max((out_width - 1) * stride[2] + filter_width - in_width, 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        output = (pad_left, pad_right, pad_top, pad_bottom, pad_forward, pad_backward)
        return output
    
#Training Method with MSE Loss Function and Adam Optimizer
def train(dataloader, model, loss_fn, optimizer):
    output = []
    size = len(dataloader.dataset)
    model.train()
    batch_num = 1
    for (batch_num, batch) in enumerate(dataloader):
        print ("Batch: " + str(batch_num+1))
        video = batch.to(torch.float32)
        video = torch.permute(video, (0,2,1,3,4))
        # Output of Autoencoder
        reconstructed = model(video)

        #Calculate Loss
        loss = loss_fn(reconstructed, video)

        #Backpropagate
        # The gradients are set to zero, the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        losses.append(loss.item())

        if (batch_num == 31):
            break
    return reconstructed


        
#Test Method to test Accuracy of Model's Predictions
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    num_batches = len(dataloader)
    tot_loss = 0

    for video in dataloader:
        video.to(device)

        reconstructed = model(video)

        #Returns val of tensor as int and adds to total loss
        tot_loss += loss_fn(reconstructed, video).item()

    tot_loss = tot_loss/num_batches
    print ("Loss: " + tot_loss)

def show():
    #Display Reconstructed vs Original

    fig, ax = plt.subplots()
    ims = []
    sm1 = video[0]
    sample1 = sm1[0]
    sample1 = sample1.detach().numpy()
    sm2 = reconstructed[0]
    sample2 = sm2[0]
    sample2 = sample2.detach().numpy()
    sample_list = [sample1, sample2]
    
    for i in range(2):
        sample = sample_list[i]
        for j in range(17):
            im = ax.imshow(sample[j], animated=True)
            if j == 0:
                ax.imshow(sample[j])  # show an initial one first
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval = 20, blit = True, repeat_delay = 1000)


    plt.show()



####Main Running Function 

in_channels = 1  # Assuming grayscale video frames
model = Autoencoder(in_channels)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, betas=(0.9,0.999))

epochs = 3
losses = []
for epoch in range(epochs):
    print ("Epoch: " + str(epoch+1))
    train(train_loader, model, loss_fn, optimizer)
"""
# Plotting the loss function
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
"""
