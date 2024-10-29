#Copied from Jupyter notebook

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import DDPMPipeline, DDPMScheduler
import numpy as np
import matplotlib.pyplot as plt

import kagglehub

# Re-download the dataset
path = kagglehub.dataset_download("thomasqazwsxedc/alphabet-characters-fonts-dataset")
print("Path to dataset files:", path)

import os

# Check if the file exists after re-downloading
npz_path = f"{path}/thomasqazwsxedc/alphabet-characters-fonts-dataset/versions/2/character_fonts (with handwritten data).npz"
if os.path.exists(npz_path):
    print("The .npz file is available.")
else:
    print("The .npz file could not be found.")

import numpy as np

# Load the dataset
data = np.load(npz_path)

# Extract images and labels
images = data['images']
labels = data['labels']

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Define the preprocessing step
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),  # Resize to 32x32 if needed by the model
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class AlphabetDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Create dataset and dataloader
alphabet_dataset = AlphabetDataset(images, labels, transform=preprocess)
dataloader = DataLoader(alphabet_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True) #Approx 40 mins to train on A100

from diffusers import DDPMPipeline, DDPMScheduler

# Initialize the model
model = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")

# Set up the scheduler
scheduler = DDPMScheduler.from_config(model.scheduler.config)

# Set the model to training mode
model.unet.train()


from torch.optim import Adam

# Move the UNet model to the GPU
model.unet.to("cuda")

# Define the optimizer using the UNet model's parameters
optimizer = Adam(model.unet.parameters(), lr=1e-4)


# Main Training Loop

import torch.nn.functional as F
import torch
from torch.optim import Adam
from torch.amp import GradScaler, autocast  # Updated AMP import
from diffusers import DDPMPipeline, DDPMScheduler
import shutil
from google.colab import files

# Initialize scheduler, optimizer, and AMP scaler
scheduler = DDPMScheduler.from_config(model.scheduler.config)
optimizer = Adam(model.unet.parameters(), lr=1e-4)
scaler = GradScaler('cuda')  # Updated AMP syntax

# Set the model to training mode
model.unet.train()

# Define the number of epochs, gradient accumulation steps, and gradient clipping threshold
num_epochs = 5
accumulation_steps = 4  # Number of steps to accumulate gradients
max_grad_norm = 1.0     # Gradient clipping threshold to prevent 'nan' values

# Training Loop with AMP, Gradient Accumulation, and Gradient Clipping
for epoch in range(num_epochs):
    total_loss = 0
    optimizer.zero_grad()  # Ensure gradients are zeroed out at the start

    for i, batch in enumerate(dataloader):
        images, _ = batch  # Ignore labels for now
        images = images.to("cuda")

        # Check and expand channels if images are grayscale
        if images.shape[1] == 1:  # If the batch is 1-channel
            images = images.expand(-1, 3, -1, -1)  # Expand to 3 channels

        # Sample random noise to add to images
        noise = torch.randn_like(images).to("cuda")
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (images.shape[0],), device="cuda").long()

        # Add noise to the images (forward diffusion process)
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        # Forward pass with AMP enabled
        with autocast('cuda'):
            noise_pred = model.unet(noisy_images, timesteps).sample
            loss = F.mse_loss(noise_pred, noise) / accumulation_steps  # Scale loss for accumulation

        # Check for NaN loss and reset gradients if NaN is detected
        if torch.isnan(loss):
            print("NaN loss detected. Skipping this batch.")
            optimizer.zero_grad()
            continue

        # Backward pass with AMP
        scaler.scale(loss).backward()

        # Gradient accumulation step with gradient clipping
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_grad_norm)  # Clip gradients

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  # Reset gradients for the next accumulation

        total_loss += loss.item() * accumulation_steps  # Scale back for reporting

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

# Save the trained model after training is complete
model_save_path = "/content/trained_model"
model.unet.save_pretrained(model_save_path)

# Zip the model directory to prepare it for download
shutil.make_archive(model_save_path, 'zip', model_save_path)
