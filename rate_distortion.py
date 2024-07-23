import matplotlib.pyplot as plt
import os
from train import train
from test import test
import time
import datetime
from datetime import timedelta

def train_rate_distortion(train_loader, test_loader, model_name, codebook_length, device, batch_size):
    
    is_show = False
    epochs = 1
    model_exist = False
    first = True
    last_loss = 1000000
    curr_loss = 5
    print ("Training L = " + str(codebook_length))
    model_codename = model_name + str(codebook_length)
    overfit = False
    #train(train_loader, model_codename, codebook_length, device, model_exist, is_show, epochs, batch_size)
    #print("Initial Training Loop Done")
    while (abs(last_loss-curr_loss)/last_loss > 0.01 and overfit == False):
        start = time.perf_counter()
        if (first):
            files = os.scandir('models')
            for file in files:
                if (model_codename == file.name):
                    model_exist = True
                    print("Model Found")
                    break
            first = False
        else:
            model_exist = True
            last_loss = curr_loss
        train_loss = train(train_loader, model_codename, codebook_length, device, model_exist, is_show, epochs, batch_size)
        curr_loss = test(test_loader, model_codename, codebook_length, device, is_show, batch_size)
        if (curr_loss > last_loss and curr_loss - last_loss > 0.1):
            overfit = True
        print ("Epoch Done. ", end = "")
        print("Train Loss: " + str(train_loss))
        print("Validation Loss: " + str(curr_loss))
        end = time.perf_counter()
        print("Time Elapsed: " + str(timedelta(seconds = end-start)))
        
    
def show_rate_distortion(test_loader, model_name, codebook_length, device, batch_size):
    is_show = False
    
    model_codename = model_name + str(codebook_length)
    print("Model Name: " + model_name)
    loss = test(test_loader, model_codename, codebook_length, device, is_show, batch_size) 
    return loss


