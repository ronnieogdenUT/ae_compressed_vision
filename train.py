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
import os
from show import show

def train_epoch(dataloader, model, loss_fn, optimizer, device, train_batches, is_show):
    #Initialize Vars
    tot_loss = 0
    

    #Setting Model Setting to Train
    model.train()

    #Iterating Through Dataloader
    for (batch_num, batch) in enumerate(dataloader):
        batch = batch.to(device)
        #print("Batch: " + str(batch_num + 1))

        #Convert Int8 Tensor to NP-usable Float32
        batch = batch.to(torch.float32)

        #Shift Tensor from size (16,20,1,64,64) to size(16,1,20,64,64)
        batch = torch.permute(batch, (0,2,1,3,4))

        # Output of Autoencoder
        reconstructed = model(batch)
        
        if (is_show and batch_num == 0):
            original_batch = torch.permute(batch, (0,2,1,3,4))
            reconstructed_batch = torch.permute(reconstructed, (0,2,1,3,4))

        #Calculate Loss
        loss = loss_fn(reconstructed, batch)
        int_loss = loss.item()
        tot_loss = tot_loss + int_loss

        #Backpropagate
        # The gradients are set to zero, the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Setting Number of Batches per Epoch
        if ((batch_num  + 1) == train_batches):
            #Cleanup
            batch = None
            reconstructed = None
            loss = None
            avg_loss = tot_loss/train_batches
            if is_show:
                return avg_loss, original_batch, reconstructed_batch
            else:
                return avg_loss
            break


def train(dataloader, model_name, codebook_length, device, model_exist, is_show):
    in_channels = 1
    epochs = 5
    losses = []
    train_batches = 16
    original_batches = []
    reconstructed_batches = []
    model_name = model_name + '.pth'

    model = Autoencoder(in_channels, codebook_length, device).to(device) #Intialize Model
    if (model_exist == True):
        model.load_state_dict(torch.load(model_name))

    loss_fn = nn.MSELoss() #Intialize Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, betas=(0.9,0.999)) #Intialize Adam Optimizer for model weights
    
    for epoch in range(epochs):
        print ("Epoch: " + str(epoch+1), end = "")
        if is_show:
            avg_loss, orig_batch, recon_batch = train_epoch(dataloader, model, loss_fn, optimizer, device, train_batches, is_show)
            original_batches.append(orig_batch)
            reconstructed_batches.append(recon_batch)
        else:
            avg_loss = train_epoch(dataloader, model, loss_fn, optimizer, device, train_batches, is_show)
        print ("  |   Average Loss per Batch = " + str(avg_loss))
        losses.append(avg_loss)
        gc.collect()
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), model_name)
    print("Saved Model")

    model_exist = True
    if is_show:
        show(original_batches, reconstructed_batches)
        # Plotting the loss function
        plt.plot(losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

    return avg_loss