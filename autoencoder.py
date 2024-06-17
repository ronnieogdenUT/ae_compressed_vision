import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
import torch.nn.functional as f
import numpy as np
import torch.nn as nn
import matplotlib.animation as animation
import math

#Import MovingMNIST Training Dataset
train_data = datasets.MovingMNIST(
    root = "./data", 
    download = True
)

#Import MovingMNIST Test Dataset
# test_data = datasets.MovingMNIST(
#     root = "./data", 
#     split = "test", 
#     download = True
# )


#Initialize Dataloader over training data
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size = batch_size, 
    shuffle = True
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
    #Initialize Vars
    train_batches = 32 #Amount of Batches to work through per epoch
    tot_loss = 0

    #Setting Model Setting to Train
    model.train()

    #Iterating Through Dataloader
    for (batch_num, batch) in enumerate(dataloader):
        batch = batch.to(device)
        #print ("Batch: " + str(batch_num+1))

        #Convert Int8 Tensor to NP-usable Float32
        batch = batch.to(torch.float32)

        #Shift Tensor from size (32,20,1,64,64) to size(32,1,20,64,64)
        batch = torch.permute(batch, (0,2,1,3,4))

        # Output of Autoencoder
        reconstructed = model(batch)

        #Calculate Loss
        loss = loss_fn(reconstructed, batch)
        tot_loss = tot_loss + loss

        #Backpropagate
        # The gradients are set to zero, the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print (f"Loss: {loss}")

        #Setting Number of Batches per Epoch
        if ((batch_num  + 1) == train_batches):
            return reconstructed, tot_loss
            break
    return tot_loss


#Test Method to test Accuracy of Model's Predictions
def test(dataloader, model, loss_fn):
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


#Display Reconstructed vs Original
def show(batches_list):

    #Define Plot + Axes + Initial Frame List
    fig, ax = plt.subplots()
    ims = []

    #Iterates through Batches, , and then through frames in video
    for batch in batches_list:

        #take first video in batch
        sample_video = batch[0]

        for frame in sample_video:
            #convert frame tensor to image(frame[0] because size is (1,64,64))
            im = ax.imshow(frame[0].cpu().detach().numpy(), animated = True)

            #append frame to frame list
            ims.append([im])

        #create video from frame list
        ani = animation.ArtistAnimation(fig, ims, interval = 50, repeat_delay = 1000)

    #display video
    plt.show()


#Main Function 
def main():
    model_exist = False
    is_train = True
    if (is_train):
        in_channels = 1  # Assuming grayscale video frames
        epochs = 32
        losses = []
        batches_list = []

        model = Autoencoder(in_channels).to(device) #Intialize Model
        if (model_exist):
            model.load_state_dict(torch.load("model.pth"))
        
        loss_fn = nn.MSELoss() #Intialize Loss Function
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, betas=(0.9,0.999)) #Intialize Adam Optimizer

        #Uses Trainloader to Run Videos through model and appends first batch of every epoch to batches_list
        for epoch in range(epochs):
            print ("Epoch: " + str(epoch+1), end = "")
            
            if epoch < (epochs-1):
                epoch_loss = train(train_loader, model, loss_fn, optimizer)
                print ("  |   Loss = " + epoch_loss)
            else:
                reconstructed, epoch_loss = train(train_loader, model, loss_fn, optimizer)
                print ("  |   Loss = " + epoch_loss)
                reconstructed = torch.permute(reconstructed, (0,2,1,3,4))
                batches_list.append(reconstructed)
            losses.append(epoch_loss)


        torch.save(model.state_dict(), "model.pth")
        print("Saved Model")

        #Calls show function
        show(batches_list)

        # Plotting the loss function
        plt.plot(losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

    #Test Function
    else:
        in_channels = 1  # Assuming grayscale video frames
        epochs = 32
        losses = []
        batches_list = []

        model = Autoencoder(in_channels).to(device) #Intialize Model
        if (model_exist):
            model.load_state_dict(torch.load("model.pth"))
        
        loss_fn = nn.MSELoss() #Intialize Loss Function
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, betas=(0.9,0.999)) #Intialize Adam Optimizer

        #Take First Batch from Original Video, append to batches_list
        for batch in train_loader:
            batches_list.append(batch)
            break

        #Uses Trainloader to Run Videos through model and appends first batch of every epoch to batches_list
        for epoch in range(epochs):
            print ("Epoch: " + str(epoch+1), end = "")
            
            if epoch < (epochs-1):
                epoch_loss = test(train_loader, model, loss_fn)
                print ("  |   Loss = " + epoch_loss)
            else:
                reconstructed, epoch_loss = train(train_loader, model, loss_fn)
                print ("  |   Loss = " + epoch_loss)
                reconstructed = torch.permute(reconstructed, (0,2,1,3,4))
                batches_list.append(reconstructed)
            losses.append(epoch_loss)


        torch.save(model.state_dict(), "model.pth")
        print("Saved Model")

        #Calls show function
        show(batches_list)

        # Plotting the loss function
        plt.plot(losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
        

#Call Main Function
main()