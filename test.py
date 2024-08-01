import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
import torch.nn.functional as f
import numpy as np
import torch.nn as nn
import matplotlib.animation as animation
import math
from torch.utils.data.sampler import SubsetRandomSampler
#from pytorch_msssim import ms_ssim
import gc
import time
import autoencoder
from autoencoder import Autoencoder


#Test Method to test Accuracy of Model's Predictions
def test(dataloader, model_name, codebook_length, device):
    with torch.no_grad(): 
        num_testBatches = 20 #How Many Batches to Run Through, Max = 2,000
        num_videos_show = 10 #How many Videos to Show at End
        num_every_video = num_testBatches/num_videos_show #Take a batch per # batches
        original_batches = []
        reconstructed_batches = []
        tot_loss = 0
        in_channels = 1  # Assuming grayscale video frames
        losses = []

        model = Autoencoder(in_channels, codebook_length, device).to(device) #Intialize Model

        model.load_state_dict(torch.load(model_name))
        
        loss_fn = nn.MSELoss() #Intialize Loss Function
        # loss_fn = ms_ssim  # Initialize Loss Function

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
            # loss = 1 - ms_ssim(reconstructed, batch, data_range=1.0, size_average=True).item()
            # tot_loss = tot_loss + loss
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
        
        print("Average Loss per Batch: " + str(avg_loss))

        #Calls show function
        show(original_batches, reconstructed_batches)


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