# ae_compressed_vision
### Authored: Gaurav Mitra, 6/13/2024

This is a general overview of the autoencoder.py file.

Description: This file containers a neural network model and the capability to train it on the MNIST dataset.

## Imports:
    torch 
        nn
        nn.functional as f
    torchvision
        datasets
        transforms.functional as tf
    matplotlib 
        pyplot as plt
        animation
    numpy - np
    math

## Data:
### Datasets: MovingMNIST
        Description: A set of 10,000 videos with 20 frames each in grayscale of two numbers moving around a 64x64 screen.
        Size: (10,000 x 20 x 1 x 64 x 64)

### Dataloader: Trainloader
        Size: (32 x 20 x 1 x 64 x 64)
        Batch Size: 32

## Classes:
### resblock_a(torch.nn.Module)
        Purpose:
            Convolves a 128 in_channel tensor to another 128 in_channel tensor. Serves as a side channel of information with original input to help decrease loss of original information.
        Input:
            Inherits from nn.Module
        Output:
            x - convolved output
        Variables:
            tensor x - 6D tensor with 128 channels
    
### resblock_b(torch.nn.Module)
        Purpose:
            Runs resblock_a 3 times
        Input:
            Inherits from nn.Module
        Output:
            x - convolved output
        Variables:
            tensor x - 6D tensor with 128 channels

### resblock_c(torch.nn.Module)
        Purpose:
            Runs resblock_b 6 times
        Input:
            Inherits from nn.Module
        Output:
            x - convolved output
        Variables:
            tensor x - 6D tensor with 128 channels

### Autoencoder(torch.nn.Module)
        Purpose:
            Autoencoder Model that contains encoder + decoder. Transforms a (20,1,64,64) into (20,128,8,8) and deconvolves back to (20,1,64,64)
        Input:
            Inherits from nn.Module
        Output:
            out - convolved and transposed convolved output
        Variables:
            tensor x - 6D tensor with 128 channels

## Functions:
### def same_pad(self, x, stride, kernel)
        Purpose:
            Function that mimics TensorFlow's "padding = 'same'" parameter of convolutions.
        Input:
            Autoencoder self - object itself(inherits from autoencoder class)
            tensor x - input tensor
            tuple stride - stride =  (a,b,c)
            int kernel - length/width of square kernel
        Output:
            tuple output - padding for (left, right, top, bottom, forward, backward) with length 6
        Variables:
            list size - size of 6D tensor
            int in_height - height of frame image
            int in_width - width of frame image
            int filter_height - height of kernel
            int filter_width - width of kernel
            int out_height - output height
            int in_height - input height
            int pad_forward - padding for forward time dimension
            int pad_backward - padding for backward time dimension
            int pad_along_height - total padding for height
            int pad_along_width - total padding for width
            int pad_top - padding for top (pad_along_height/2)
            int pad_bottom - padding for bottom (pad_along_height/2)
            int pad_right - padding for right side(pad_along_width/2)
            int pad_left - padding for left side(pad_along_width/2)
            tuple output - size 6 tuple of paddings for all sides

### def train(dataloader, model, loss_fn, optimizer)
        Purpose:
            Trains Model using Adam optimizer and calculating MSE loss
        Input:
            Dataloader dataloader - dataloader object that wraps around MNIST dataset with batch_size = 32
            Autoencoder model - instance of class autoencoder
            loss_fn - MSE loss function
            optimizer - adam optimizer
        Output:
            tensor reconstructed - last batch of epoch reconstructed through model of size (32,20,1,64,64)
        Variables:
            int train_batches - amount of batches to train per epoch
            list losses - list of losses per batch
            int batch_num - iterable for batches in dataloader
            tensor batch - tensor of size (32,20,1,64,64): 32 sets of 20 frame videos
            tensor reconstructed - batch after pass through model
            int loss - loss from MSE loss calculation

    
### def test(dataloader, model, loss_fn)
        Purpose:
            Function that tests model accuracy and caluclates total loss per batch
        Input:
            Dataloader dataloader - dataloader object that wraps around MNIST dataset with batch_size = 32
            Autoencoder model - instance of class autoencoder
            loss_fn - MSE loss function
        Output:
            list loss_batches - list of loss per batch
        Variables:
            int num_batches - number of batches
            int tot_loss - total amount of loss per epoch
    
### def show(batches_list)
        Purpose:
            Displays first video in each batch in batches_list
        Input:
            list batches_list - first batch is original batch pre-training and then subsequent batch per epoch
        Output:
            N/A(display function)
        Variables:
            figure fig - matplot figure
            axes ax - axes for figure
            list ims - list of image frames
            tensor sample_video - first video in batch
            tensor im - frame tensor in sample_video
            AristAnimation ani - video of the different videos per epoch

### def main()
        Purpose:
            Currently trains model for set number of epochs and calls show() to display first video in first batch
        Input:
            N/A(main)
        Output:
            N/A(main)
        Variables:
            int in_channels - amount of in_channels for video(1 since grayscale video)
            int epochs - amount of epochs to run
            list losses - list of losses per epoch
            list batches_list - first batch is original batch pre-training and then subsequent batch per epoch
            Autoencoder model - instance of autoencoder class, training model
            nn loss_fn - MSE loss function
            torch.optim optimizer - adam optimizer
            tensor reconstructed - last batch of epoch reconstructed through model of size (32,20,1,64,64)
        

