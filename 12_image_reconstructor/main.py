#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
""" 
Image Reconstructor from the latent discrete vector with the MNIST dataset.
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Utility functions
print("Cell: 0")
print("Utility functions loaded.")

def cell_counter(n:int) -> None:
    print("Cell:", n)

import time    
def time_count_start() -> float:
    return time.time()

def time_count_end(start_time: float) -> None:
    final_time = time.time()
    print(f"Cell time: {final_time - start_time:.3f} seconds")

time_start: float    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Library imports

time_start = time_count_start()
cell_counter(1)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

print("Libraries imported.")

time_count_end(time_start)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Hyperparameters and configurations
time_start = time_count_start()
cell_counter(2)

latent_dim:int = 10
hidden_dim_1:int = 128
hidden_dim_2:int = 256
hidden_dim_3:int = 512

l_rate:float = 0.001
batch_size:int = 64
epochs:int = 2

time_count_end(time_start)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Device configuration

time_start = time_count_start()
cell_counter(3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")    

time_count_end(time_start)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Data loading and preprocessing

time_start = time_count_start()
cell_counter(4)

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print("Data loaded.")
time_count_end(time_start)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Funcion ONE-HOT ENCODER
time_start = time_count_start()
cell_counter(5)

def one_hot_encode(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.eye(num_classes, device=device)[label].to(device)

test_label = torch.randint(0, 10, (1,))
num_classes = latent_dim   
one_hot_vector = one_hot_encode(test_label, num_classes)
print(f"One-hot encoded vector for label {test_label.item()}: \n{one_hot_vector}")

time_count_end(time_start)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Function to reset model weights
def reset_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        m.reset_parameters()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Decoder model definition
time_start = time_count_start()
cell_counter(6)

decoder = nn.Sequential(
    nn.Linear(latent_dim, hidden_dim_1),
    nn.ReLU(),
    nn.Linear(hidden_dim_1, hidden_dim_2),
    nn.ReLU(),
    nn.Linear(hidden_dim_2, hidden_dim_3),
    nn.ReLU(),
    nn.Linear(hidden_dim_3, 28*28),
    nn.Sigmoid(),
    nn.Unflatten(1, (1, 28, 28)) # Reshape output to (1, 28, 28). This 1 is for compatibility with pytorch dataloader
).to(device)

print("Decoder model defined.")
time_count_end(time_start)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Model definition

time_start = time_count_start()
cell_counter(7) 

class ImageReconstructor(nn.Module):
    def __init__(self, decoder: nn.Module, latent_dim: int):
        super(ImageReconstructor, self).__init__()
        self.decoder = decoder
        self.latent_dim = latent_dim

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        # One-hot encode the labels
        one_hot_labels = one_hot_encode(labels, self.latent_dim)
        # Decode the one-hot vectors to reconstruct images
        reconstructed_images = self.decoder(one_hot_labels)
        return reconstructed_images

model = ImageReconstructor(decoder, latent_dim).to(device)
print("Model defined.")
time_count_end(time_start)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Loss function and optimizer

time_start = time_count_start()
cell_counter(8)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=l_rate)
print("Loss function and optimizer defined.")

time_count_end(time_start)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Function to plot loss during train
time_start = time_count_start()
cell_counter(9)

def generate_loss_plot(train_steps: int, loss_vector: list) -> None:
    #plt.figure(figsize=(10, 5))
    plt.plot(range(train_steps), loss_vector, label='Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.legend()
    plt.grid()
    plt.show()
    
print("Plot function defined.")
time_count_end(time_start)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Funtion to visualize reconstructed images
cell_counter(10)
time_start = time_count_start()

def visualize_reconstructions(model: nn.Module) -> None:
    model.eval()
    with torch.no_grad():
        for label in range(latent_dim):
            label_tensor = torch.tensor([label], device=device)
            output = model(label_tensor)
            
            plt.imshow(output.cpu().squeeze(), cmap='gray')
            plt.title(f"Expected draw: {label}")
            plt.axis('off')
            plt.show()
            

time_count_end(time_start)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Training loop

time_start = time_count_start()
cell_counter(11)    

train_losses = []
train_steps = 0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(labels)
        loss = criterion(outputs, images)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        train_losses.append(loss.item())
        train_steps += 1
        
        if (batch_idx + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    visualize_reconstructions(model)
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")

generate_loss_plot(train_steps, train_losses)   
time_count_end(time_start)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.apply(reset_weights)



# %%
