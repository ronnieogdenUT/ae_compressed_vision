import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from autoencoder import Autoencoder
import os
from show import show

def train_epoch(dataloader, model, loss_fn, optimizer, device, train_batches, is_show):
    #Initialize Vars
    avg_loss = 0

    #Iterating Through Dataloader
    for (batch_num, batch) in enumerate(dataloader):
        #print ("Batch: " + str(batch_num), end="")
        #Convert Int8 Tensor to NP-usable Float32
        batch = batch.to(device, dtype = torch.float32)

        #Shift Tensor from size (B,20,1,64,64) to size(B,1,20,64,64)
        batch = torch.permute(batch, (0,2,1,3,4))

        # Output of Autoencoder
        reconstructed = model(batch)
        
        if (is_show and batch_num == 0):
            original_batch = torch.permute(batch, (0,2,1,3,4))
            reconstructed_batch = torch.permute(reconstructed, (0,2,1,3,4))

        #Calculate Loss
        loss = loss_fn(reconstructed, batch)
        avg_loss += loss.item()/train_batches
        #print ("Loss: " + str(loss.item()))

        #Backpropagate
        # The gradients are set to zero, the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #Setting Number of Batches per Epoch
        if ((batch_num  + 1) == train_batches):
            if is_show:
                return avg_loss, original_batch, reconstructed_batch
            else:
                return avg_loss
            break


def train(dataloader, model_name, codebook_length, device, model_exist, is_show, epochs, batch_size):
    in_channels = 1
    losses = []
    train_batches = int(8000/batch_size)
    original_batches = []
    reconstructed_batches = []
    model_name = model_name + '.pth'
    model_path = os.path.join('models', model_name)

    model = Autoencoder(in_channels, codebook_length, device, batch_size).to(device) #Intialize Model
    if (model_exist == True):
        model.load_state_dict(torch.load(model_path))
        print("Model Loaded")

    loss_fn = nn.MSELoss() #Intialize Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas=(0.9,0.999)) #Intialize Adam Optimizer for model weights

    model.train()
    
    for epoch in range(epochs):
        if is_show:
            print ("Epoch: " + str(epoch+1), end = "")
            avg_loss, orig_batch, recon_batch = train_epoch(dataloader, model, loss_fn, optimizer, device, train_batches, is_show)
            original_batches.append(orig_batch)
            reconstructed_batches.append(recon_batch)
            print ("  |   Average Loss per Batch = " + str(avg_loss))
        else:
            avg_loss = train_epoch(dataloader, model, loss_fn, optimizer, device, train_batches, is_show)
        losses.append(avg_loss)
        torch.save(model.state_dict(), model_path)
        print("Saved Model")

    #del model
    if is_show:
        show(original_batches, reconstructed_batches)
        # Plotting the loss function
        plt.plot(losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
    else:
        return avg_loss