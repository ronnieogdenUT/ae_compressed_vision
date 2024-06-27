import torch
from torch import nn
import math
from torch import abs

def quantize(z, centroids):
    tau = 10**7
    #z output from encoder as B x D x Channels x L x W
    z_shape = z.shape

    #Compute Distances
    distances = (abs(z - centroids))**2
    
    #Get closest centroid
    Qd = torch.argmin(distances, dim=1)
    total_sum = torch.sum(math.exp(-tau*abs(z-centroids)))
    Qs = torch.softmax(torch.sum(((math.exp(-tau*abs(z-centroids)))/total_sum)*centroids))

    z_bar = (Qd - Qs).detach() + Qs

    return z_bar

    #One Hot Encoding
