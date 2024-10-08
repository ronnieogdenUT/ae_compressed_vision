import torch
import torch.nn.functional as f
import torch.nn as nn
import math

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
    def __init__(self, in_channels, codebook_length, device, batch_size):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        #z output from encoder as B x D x Channels x L x W
        #Initialize centroids to L x 1
        centroids = torch.ones((codebook_length,1), dtype = torch.float32, device = device)
        torch.nn.init.kaiming_uniform_(centroids, mode="fan_in", nonlinearity="relu")
        centroids = torch.squeeze(centroids)
        self.centroids = nn.Parameter(centroids)
        self.codebook_length = codebook_length

        #Encoder
        self.encoderConv1 = torch.nn.Conv3d(in_channels, 64, 5, stride=(1,2,2))
        self.encoderBn1 = torch.nn.BatchNorm3d(64)
        self.encoderConv2 = torch.nn.Conv3d(64, 128, 5, stride=(1,2,2))
        self.encoderBn2 = torch.nn.BatchNorm3d(128)
        self.encoderConv3 = torch.nn.Conv3d(128, 32, 5, stride=(1,2,2))
        self.encoderBn3 = torch.nn.BatchNorm3d(32)  

        self.resblock_c = resblock_c()
        	
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

        del quantized_x
        
        return x
    
    #Calculates Padding(Mimics Tensor Flow padding = 'same')
    def same_pad(self, x, stride, kernel):
        size = x.size()

        if len(size) == 5: #Batch of Videos: Batch x Channel x Time x Length x Width
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
        else: #Video(Batch of Frames): Batch x Channel x Length x Width
            in_height = size[3]
            in_width = size[2]
            filter_height = kernel
            filter_width = kernel

            out_height = math.ceil(float(in_height) / float(stride[1]))
            out_width  = math.ceil(float(in_width) / float(stride[2]))
            
            #Regular 2D Padding: Top, Bottom, Left, Right
            pad_along_height = max((out_height - 1) * stride[1] + filter_height - in_height, 0)
            pad_along_width = max((out_width - 1) * stride[2] + filter_width - in_width, 0)
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
            output = (pad_left, pad_right, pad_top, pad_bottom)
            return output
    
    def quantize(self, x):
        quantized_shape = list(x.shape)
        quantized_shape.append(self.codebook_length)

        distances = torch.ones(quantized_shape, device = self.device)

        for i in range(self.codebook_length):
            distances[...,i] = abs(x - self.centroids[i])

        Qs = torch.softmax(distances, dim = -1)
        Qh = torch.min(distances, dim = -1, keepdim=True)[0]

        Qs = torch.sum(Qs, dim=-1)
        Qh = torch.sum(Qh, dim=-1)

        quantized_x = Qs + (Qh.detach() - Qs.detach())

        return quantized_x
