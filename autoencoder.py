import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
import numpy as np
import torch.nn as nn
import matplotlib.animation as animation


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
    dataset = train_data, #####fix this later
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

#Calculate Padding Size
#padding = 2*(64 - 1) - 64 + 2  = 64
#padding = Stride * (input Vol Size - 1) - input Vol Size + kernel size

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

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, 64, 5, stride=(1,2,2), padding='valid'),
			torch.nn.BatchNorm3d(64),
			torch.nn.ReLU(),
            torch.nn.Conv3d(64, 128, 5, stride=(1,2,2), padding='valid'),
			torch.nn.BatchNorm3d(128),
			torch.nn.ReLU(),
            resblock_c(),
            torch.nn.Conv3d(128, 32, 5, stride=(1,2,2), padding='valid'),
            torch.nn.BatchNorm3d(32)
		)      
        
        # Figure out if resblock_c needs to be a transposed version... I think they are the same here		
        self.decoder = torch.nn.Sequential(
			torch.nn.ConvTranspose3d(32, 128, 3, stride=(1,2,2), padding= 0),
            torch.nn.BatchNorm3d(128),
			torch.nn.ReLU(),
            resblock_c(),
            torch.nn.ConvTranspose3d(128, 64, 5, stride=(1,2,2), padding= 0),
            torch.nn.BatchNorm3d(64),
			torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(64, in_channels, 5, stride=(1,2,2), padding= 0),
            torch.nn.BatchNorm3d(in_channels)
		)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    

in_channels = 1  # Assuming video frames
model = Autoencoder(in_channels)


epochs = 2
outputs = []
losses = []
for epoch in range(epochs):
    
    for video in train_loader:

        # print(video.dtype)
        video = video.to(torch.float32)
        video = torch.permute(video, (0,2,1,3,4))
        # video = tf.cast(video, tf.float32)
        print(video.dtype)
        print(video.size())

        # Output of Autoencoder
        reconstructed = model(video)
        
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

        # The gradients are set to zero, the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        losses.append(loss.item())
    outputs.append((epoch, video, reconstructed))

# Plotting the loss function
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

