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
batch_size = 32
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
    def __init__(self, in_channels, codebook_length):
        super().__init__()
        self.tau = 10
        #z output from encoder as B x D x Channels x L x W
        #Initialize centroids to 32x32x20x8x8xL
        self.centroids = nn.Parameter(torch.ones([codebook_length, 32,32,20,8,8], dtype = torch.float32).to("cpu"))
        torch.nn.init.kaiming_uniform_(self.centroids, mode="fan_in", nonlinearity="relu")
        self.codebook_length = codebook_length

        #Encoder
        self.encoderConv1 = torch.nn.Conv3d(in_channels, 64, 5, stride=(1,2,2))
        self.encoderBn1 = torch.nn.BatchNorm3d(64)
        self.encoderConv2 = torch.nn.Conv3d(64, 128, 5, stride=(1,2,2))
        self.encoderBn2 = torch.nn.BatchNorm3d(128)
        self.encoderConv3 = torch.nn.Conv3d(128, 32, 5, stride=(1,2,2))
        self.encoderBn3 = torch.nn.BatchNorm3d(32)  

        #self.resblock_c = resblock_c()
        
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
        #x = resblock_c(x)

        x = f.pad(x, self.same_pad(x, stride, 5))
        x = self.encoderConv3(x)
        x = self.encoderBn3(x)

        #Quantize
        quantized_x = self.quantize(x)

        #Decoder
        x = self.decoderConv1(quantized_x)
        x = self.decoderBn1(x)
        x = f.relu(x)
        #x = resblock_c(x)

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
    
    def quantize(self, x):
        #Compute Distances
        centroids = self.centroids
        #Get closest centroid
        #Qd = torch.argmin(distances, dim=1)

        total_sum = torch.ones(centroids.shape)

        #print(total_sum)

        #Calculate Qs, size(code_length x 32 x 32 x 20 x 8 x 8)
        #print("Centroids: " + str(numpy_centroids.shape))
        Qs = torch.ones(centroids.shape)
        for i in range(self.codebook_length):
            distance = (abs(x - centroids[i, :]))
            Qs[i] = torch.exp(-self.tau*distance)

        quantized_x = (Qs * centroids)/torch.sum(Qs)

        #Multiply Qs with centroids to get closest Codebook Value
        #Multiplies Qs(L x 32 x 32 x 20 x 8 x 8) and centroids(L x 32 x 32 x 20 x 8 x 8) and converts to tensor

        # print(quantized_x.shape)
        # print(quantized_x[:, 1, 1, 1, 1, 1])

        #Now we have the L x 32 x 32 x 20 x 8 x 8, which should entirely be one codebook value with 
        quantized_x = torch.sum(quantized_x, dim=0)

        #Reduced down to the one codebook value
        print(quantized_x)

        #Full Quant(Not implemented)
        #z_bar = (Qd - Qs).detach() + Qs

        return quantized_x
    

#Training Method with MSE Loss Function and Adam Optimizer
def train(dataloader, model, loss_fn, optimizer):
    #Initialize Vars
    train_batches = 1 #Amount of Batches to work through per epoch
    tot_loss = 0

    #Setting Model Setting to Train
    model.train()

    #Iterating Through Dataloader
    for (batch_num, batch) in enumerate(dataloader):
        #print ("Batch: " + str(batch_num+1))
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
        int_loss = loss.item()
        #print("Batch Loss: " + str(int_loss))
        tot_loss = tot_loss + int_loss

        #Backpropagate
        # The gradients are set to zero, the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print (f"Loss: {loss}")

        #Setting Number of Batches per Epoch
        if ((batch_num  + 1) == train_batches):
            return tot_loss
            break


#Test Method to test Accuracy of Model's Predictions
def test(dataloader, model, loss_fn):
    num_testBatches = 20 #How Many Batches to Run Through, Max = 2,000
    model.eval()
    num_videos_show = 10 #How many Videos to Show at End
    num_every_video = num_testBatches/num_videos_show #Take a batch per # batches
    original_batches = []
    reconstructed_batches = []
    tot_loss = 0

    for (batch_num, batch) in enumerate(dataloader):
        print ("Batch: " + str(batch_num))
        batch = batch.to(device)

        #Convert Int8 Tensor to NP-usable Float32
        batch = batch.to(torch.float32)

        #Shift Tensor from size (32,20,1,64,64) to size(32,1,20,64,64)
        batch = torch.permute(batch, (0,2,1,3,4))

        # Output of Autoencoder
        reconstructed = model(batch)

        #Calculate Loss
        loss = loss_fn(reconstructed, batch).item()
        tot_loss = tot_loss + loss

        #Every "num_videos_show" batches append first vid: originial and reconstructed
        if (batch_num % num_every_video == 0):
            original_batches.append(torch.permute(batch, (0,2,1,3,4)))
            reconstructed_batches.append(torch.permute(reconstructed, (0,2,1,3,4)))


        #Setting Number of Batches to Test
        if ((batch_num  + 1) == num_testBatches):
            break
    avg_loss = tot_loss/num_testBatches
    return original_batches, reconstructed_batches, avg_loss


#Display Reconstructed vs Original
def show(original_batchList, reconstructed_batchList):
    #Define Plot + Axes + Initial Frame List
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ims = []

    #Iterates through Batches, , and then through frames in video
    for batch in range(len(original_batchList)):

        #Get Batches
        original_batch = original_batchList[batch]
        reconstructed_batch = reconstructed_batchList[batch]
        #Take first video from batches
        original_sample_video = original_batch[0]
        reconstructed_sample_video = reconstructed_batch[0]

        for original_frame, reconstructed_frame in zip(original_sample_video, reconstructed_sample_video):
            #convert frame tensor to image(frame[0] because size is (1,64,64))
            im1 = ax1.imshow(original_frame[0].cpu().detach().numpy(), animated = True)
            im2 = ax2.imshow(reconstructed_frame[0].cpu().detach().numpy(), animated = True)

            #append frame to frame list
            ims.append([im1, im2])

    #create video from frame list
    ani = animation.ArtistAnimation(fig, ims, interval = 50, repeat_delay = 1000)

    #display video
    plt.show()


#Main Function 
def main(is_train, model_name, codebook_length):
    if (is_train):
        in_channels = 1  # Assuming grayscale video frames
        epochs = 1
        losses = []
        batches_list = []

        model = Autoencoder(in_channels, codebook_length).to(device) #Intialize Model
        if (model_exist == True):
            model.load_state_dict(torch.load(model_name))

        
        loss_fn = nn.MSELoss() #Intialize Loss Function
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, betas=(0.9,0.999)) #Intialize Adam Optimizer for model weights

        #Uses Trainloader to Run Videos through model and appends first batch of every epoch to batches_list
        for epoch in range(epochs):
            print ("Epoch: " + str(epoch+1), end = "")
            epoch_loss = train(train_loader, model, loss_fn, optimizer)
            print ("  |   Loss = " + str(epoch_loss))
            losses.append(epoch_loss)


        torch.save(model.state_dict(), model_name)
        print("Saved Model")

        # Plotting the loss function
        plt.plot(losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

    #Test Function
    else:
        in_channels = 1  # Assuming grayscale video frames
        losses = []
        batches_list = []

        model = Autoencoder(in_channels, codebook_length).to(device) #Intialize Model

        model.load_state_dict(torch.load(model_name))
        
        loss_fn = nn.MSELoss() #Intialize Loss Function

        #Uses TestLoader to Run Videos through model
        original_list, reconstructed_list, avg_loss = test(test_loader, model, loss_fn)

        print("Average Loss per Batch: " + str(avg_loss))

        #Calls show function
        show(original_list, reconstructed_list)

        

#Call Main Function
model_name = "modelQuant.pth"
model_exist = False
is_train = True
codebook_length = 20
main(is_train, model_name, codebook_length)