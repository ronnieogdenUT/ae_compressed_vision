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

### def train(dataloader, model_name, codebook_length, device, model_exist)
        Purpose:
            Trains Model
            Initializes Model
            If model_exist is true, load parameters of model from file
            Initialize loss function and optimizer
            Iterates through int epochs and calls train_epoch
            Appends epoch_loss to list losses
            Displays graph of epoch_loss over iterations
        Input:
            Dataloader dataloader - dataloader object that wraps around MNIST dataset with batch_size = 16
            str model_name - name of model to be imported or saved into a file
            int codebook_length - length of codebook/ num of values to be trained
            str device - device to train on(cuda or cpu)
            bool model_exist - whether model exists or not
        Output:
            N/A
            prints out epoch loss & loss per epoch on graph
        Variables:
            
            int train_batches - amount of batches to train per epoch
            int epochs - number of epochs to train model on (train_batches) batches
            list losses - list of losses per epoch
            int in_channels - number of channels for input(1 for grayscale)
            int epoch_loss - loss per epoch

### def train_epoch(dataloader, model, loss_fn, optimizer, device, train_batches)
        Purpose:
            Trains model
            Iterates through batches on an individual Epoch
            Moves Batch to device
            Transforms batch to model-ready input
            Calculates loss for batch and adds it to total epoch loss
            Uses Adam optimizer and MSE loss on backward pass
            If batch_num is at train_batches, return epoch loss
        Input:
            Dataloader dataloader - dataloader object that wraps around MNIST dataset with batch_size = 16
            str model_name - name of model to be imported or saved into a file
            int codebook_length - length of codebook/ num of values to be trained
            str device - device to train on(cuda or cpu)
            bool model_exist - whether model exists or not
        Output:
            int tot_loss - total loss for that epoch
        Variables:
            int batch_num - iterable for batches in dataloader
            tensor batch - tensor of size (16,20,1,64,64): 16 sets of 20 frame videos
            tensor reconstructed - batch after pass through model
            int loss - loss from MSE loss calculation
            loss_fn - MSE loss function

    
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
        

