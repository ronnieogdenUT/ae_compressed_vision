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
import gc
import autoencoder
from autoencoder import Autoencoder

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