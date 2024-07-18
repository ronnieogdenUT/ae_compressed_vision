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
import sys
from train import train
from test import test

def rate_distortion(train_loader, test_loader, model_name, codebook_length, device, model_exist):
    codebook_vals = [10, 30]
    is_show = False
    last_loss = 1000000
    curr_loss = 5
    first = True
    losses = []
    epochs = 1
    for codebook_length in codebook_vals:
        print ("Training L = " + str(codebook_length))
        model_name = model_name + str(codebook_length) + '.pth'   
        while (abs(last_loss-curr_loss)/last_loss > 0.01):
            if (first):
                files = os.scandir()
                for file in files:
                    if (model_name in file.name):
                        model_exist = True
                        print("Model Found")
                        break
                first = False
                continue
            else:
                model_exist = True
                last_loss = curr_loss
            train(train_loader, model_name, codebook_length, device, model_exist, is_show, epochs)
            curr_loss = test(test_loader, model_name, codebook_length, device, is_show)
            print ("Epoch Done. Current Loss: " + str(curr_loss))
    
def show_rate_distortion(test_loader, model_name, codebook_length, device, model_exist):
    codebook_vals = [10, 30]
    is_show = False
    losses = []
    for codebook_length in codebook_vals:
        model_name = model_name + str(codebook_length) + '.pth'   
        files = os.scandir()
        for file in files:
            if (model_name in file.name):
                model_exist = True
                print("Model Found")
                break
        loss = test(test_loader, model_name, codebook_length, device, is_show) 
        losses.append(loss)
    plt.plot(codebook_vals, losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
