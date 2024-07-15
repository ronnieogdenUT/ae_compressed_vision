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
    def __init__(self, in_channels, codebook_length, device):
        super().__init__()
        self.tau = 10
        self.device = device
        #z output from encoder as B x D x Channels x L x W
        #Initialize centroids to Lx 16 x 32 x 20 x 8 x 8
        self.centroids = nn.Parameter(torch.ones((codebook_length,1), dtype = torch.float32).to(device))
        torch.nn.init.kaiming_uniform_(self.centroids, mode="fan_in", nonlinearity="relu")
        self.codebook_length = codebook_length

        #Encoder
        self.encoderConv1 = torch.nn.Conv3d(in_channels, 64, 5, stride=(1,2,2))
        self.encoderBn1 = torch.nn.BatchNorm3d(64)
        self.encoderConv2 = torch.nn.Conv3d(64, 128, 5, stride=(1,2,2))
        self.encoderBn2 = torch.nn.BatchNorm3d(128)
        self.encoderConv3 = torch.nn.Conv3d(128, 32, 5, stride=(1,2,2))
        self.encoderBn3 = torch.nn.BatchNorm3d(32)  

        self.resblock_c = resblock_c()
        
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

        x = self.resblock_c(x)

        x = f.pad(x, self.same_pad(x, stride, 5))
        x = self.encoderConv3(x)
        x = self.encoderBn3(x)

        #Quantize
        quantized_x = self.quantize(x)

        #Decoder
        x = self.decoderConv1(quantized_x)
        x = self.decoderBn1(x)
        x = f.relu(x)

        x = self.resblock_c(x)

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
        centroids = self.centroids
        quantized_shape = (self.codebook_length, 8, 32, 20, 8, 8)

        Qs = torch.ones(quantized_shape).to(self.device)
        print(Qs.shape)
        for i in range(self.codebook_length):
            distance = (abs(x - centroids[i, :]))
            Qs[i] = torch.exp(-self.tau*distance)

        Qs = torch.permute(Qs, (1,2,3,4,5,0))
        print(Qs.shape)
        quantized_x = (Qs * centroids)/torch.sum(Qs)

        #Multiply Qs with centroids to get closest Codebook Value
        #Multiplies Qs(L x 16 x 32 x 20 x 8 x 8) and centroids(L x 16 x 32 x 20 x 8 x 8) and converts to tensor

        #Now we have the L x 16 x 32 x 20 x 8 x 8, which should entirely be one codebook value with 
        quantized_x = torch.sum(quantized_x, dim=0)

        #Reduced down to the one codebook value

        return quantized_x