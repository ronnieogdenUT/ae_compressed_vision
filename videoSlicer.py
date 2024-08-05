#Written by Rishikanth Karanam, 07/09/2024

import torch
import datetime
import cv
import numpy as np
import dv_processing as dv
from datetime import timedelta
import matplotlib.pyplot as plt
from autoencoder import Autoencoder

file_name = "inivation_front_view.aedat4"

# Open the file
reader = dv.io.MonoCameraRecording(file_name)

# Add a noise filter
resolution = reader.getEventResolution()
visualizer = dv.visualization.EventVisualizer(resolution)
filter = dv.noise.BackgroundActivityNoiseFilter(resolution, backgroundActivityDuration=datetime.timedelta(milliseconds=1))

# Create an empty list to append sliced images into as a tensor
video = []

# Initialize a lastTimestamp 
lastTimestamp = None

# Get the current timestamp
timestamp = dv.now()

# Initialize store, it will have no jobs at this time
store = dv.EventStore()

# Initialize slicer, it will have no jobs at this time
slicer = dv.EventStreamSlicer()

# define a function to print when it receives slicing
def add_frame(event_slice: dv.EventStore):
    image = visualizer.generateImage(event_slice)
    video.append(image)
    
 # Convert image to the required format for the autoencoder (e.g., torch tensor)
    image_tensor = torch.tensor(image, dtype=torch.float32).to(device)
    
    # Normalize the image tensor to the range [0, 1]
    image_tensor = image_tensor / 255.0

    # Pass the image through the autoencoder
    with torch.no_grad():
        output = Autoencoder(image_tensor)
    
    # Handle the output (e.g., convert back to numpy and visualize)
    output_image = output.squeeze().cpu().numpy()
    
    # Append the output image to the video list
    video.append(output_image)

# Register this method to be called every 33 millisecond worth of event data
slicer.doEveryTimeInterval(timedelta(milliseconds=33), add_frame)

# # Run the loop while file is still running
# while reader.isRunning():
#     # Read batch of events
#     events = reader.getNextEventBatch()
    
#     # Read a frame from the camera
#     if events is not None:
#         # Filter noise events
#         filter.accept(events)
#         filter_events = filter.generateEvents()
#         slicer.accept(filter_events)

# Convert the array to a numpy array
arr = np.array(video)

# Return the RGB values from arr and save it into a pytorch.pt file
array = torch.from_numpy(arr)
torch.save(array, 'pytorch.pt')
array = torch.load('pytorch.pt')