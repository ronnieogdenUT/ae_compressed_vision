import torch
from torchvision import datasets
import matplotlib.pyplot as plt


train_data = datasets.MovingMNIST(
    root = "./data", 
    download = True
)

# test_data = datasets.MovingMNIST(
#     root = "./data", 
#     split = "test", 
#     download = True
# )

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    dataset = train_data, 
    batch_size = batch_size, 
    shuffle = True
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class resblock_a(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_stack = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, 3, stride=(1,2,2), padding='same'),
			torch.nn.BatchNorm3d(128),
			torch.nn.ReLU(),
            torch.nn.Conv3d(128, 128, 3, stride=(1,2,2), padding='same'),
			torch.nn.BatchNorm3d(128)
        )
        
    def foward(self, x):
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
    def __init__(self, in_channels):
        super().__init__()
		
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, 64, 5, stride=(1,2,2), padding='same'),
			torch.nn.BatchNorm3d(64),
			torch.nn.ReLU(),
            torch.nn.Conv3d(64, 128, 5, stride=(1,2,2), padding='same'),
			torch.nn.BatchNorm3d(128),
			torch.nn.ReLU(),
            resblock_c(),
            torch.nn.Conv3d(128, 32, 5, stride=(1,2,2), padding='same'),
            torch.nn.BatchNorm3d(32)
		)      
        
        # Figure out if resblock_c needs to be a transposed version... I think they are the same here		
        self.decoder = torch.nn.Sequential(
			torch.nn.ConvTranspose3d(32, 128, 3, stride=(1,2,2), padding='same'),
            torch.nn.BatchNorm3d(128),
			torch.nn.ReLU(),
            resblock_c(),
            torch.nn.ConvTranspose3d(128, 64, 5, stride=(1,2,2), padding='same'),
            torch.nn.BatchNorm3d(64),
			torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(64, in_channels, 5, stride=(1,2,2), padding='same'),
            torch.nn.BatchNorm3d(in_channels)
		)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



    


# model = Autoencoder(in_channels)

# loss_function = torch.nn.MSELoss()

# optimizer = torch.optim.Adam(model.parameters(),
# 							lr = 1e-4,
# 							weight_decay = 1e-8)

# epochs = 2
# outputs = []
# losses = []
# for epoch in range(epochs):
#     for (image, _) in loader:
# 	
#         # Reshaping the image to (-1, 784)
#         image = image.reshape(-1, 28*28)
# 	
#         # Output of Autoencoder
#         reconstructed = model(image)
# 	
#         # Calculating the loss function
#         loss = loss_function(reconstructed, image)
# 	
#         # The gradients are set to zero,
#         # the gradient is computed and stored.
#         # .step() performs parameter update
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
# 	
#         # Storing the losses in a list for plotting
#         losses.append(loss)
#     outputs.append((epochs, image, reconstructed))

