import torch
import torch.nn as nn
#from pytorch_msssim import ms_ssim
from autoencoder import Autoencoder
from show import show
import os
import time
from datetime import timedelta


#Test Method to test Accuracy of Model's Predictions
def test(dataloader, model_name, codebook_length, device, is_show, batch_size):
    model_name = model_name + '.pth'
    with torch.no_grad(): 
        num_testBatches = 5 #int(2000/batch_size) #How Many Batches to Run Through, Max = 2,000
        num_videos_show = 10 #How many Videos to Show at End
        num_every_video = num_testBatches/num_videos_show #Take a batch per # batches
        original_batches = []
        reconstructed_batches = []
        avg_loss = 0
        in_channels = 1  # Assuming grayscale video frames


        model_path = os.path.join('models', model_name)
        model = Autoencoder(in_channels, codebook_length, device, batch_size).to(device) #Intialize Model
        model.load_state_dict(torch.load(model_path))

        loss_fn = nn.MSELoss() #Intialize Loss Function

        #Run model with no grad to calibrate running mean/variance for BN layers
        start = time.time()
        for (batch) in dataloader:
            reconstructed = model(batch)
            end = time.time()
            if (start-end > 60):
                break

        #Convert model to eval
        model.eval()

        #Set all layers but BN layers to model.eval() and BN on train
        # for module in model.modules():
        #     if isinstance(module, torch.nn.BatchNorm3d):  # Or BatchNorm1d depending on your model
        #         module.train()  # Keep BatchNorm layers in training mode

        for (batch_num, batch) in enumerate(dataloader):
            if is_show: print ("Batch: " + str(batch_num+1))

            #Convert Int8 Tensor to NP-usable Float32
            batch = batch.to(device, dtype = torch.float32)

            # #Shift tensor to frames first to test latency
            # batch = torch.permute(batch, (1,0,2,3,4))
            # batch = batch[0:1]
            # print(batch.shape)
            # batch = torch.permute(batch, (1,0,2,3,4))

            #Shift Tensor from size (32,20,1,64,64) to size(32,1,20,64,64)
            
            batch = torch.permute(batch, (0,2,1,3,4))

            # Output of Autoencoder
            start = time.perf_counter()
            reconstructed = model(batch)
            end = time.perf_counter()

            print("Time Elapsed: " + str(timedelta(seconds = end-start)))

            #Calculate Loss
            loss = loss_fn(reconstructed, batch).item()
            avg_loss = avg_loss + loss/num_testBatches

            #Every "num_videos_show" batches append first vid: originial and reconstructed
            if (batch_num % num_every_video == 0):
                original_batches.append(torch.permute(batch, (0,2,1,3,4)))
                reconstructed_batches.append(torch.permute(reconstructed, (0,2,1,3,4)))

            #Setting Number of Batches to Test
            if ((batch_num  + 1) == num_testBatches):
                break
        
        if is_show:
            print("Average Loss per Batch: " + str(avg_loss))
            #Calls show function
            show(original_batches, reconstructed_batches)
        else:
            return avg_loss