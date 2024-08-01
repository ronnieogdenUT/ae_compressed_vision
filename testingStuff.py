import torch

data = torch.load("pytorch.pt")
print(data.shape)

#2694 frames
#260 x 346
#3 Channels
# Can use rgb_to_grayscale