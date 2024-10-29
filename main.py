from diffusers import DDPMPipeline
import torch
import safetensors.torch

import matplotlib.pyplot as plt
from tqdm import tqdm

# Path to your model directory
model_path = "/Users/Pradeep/Desktop/Python/diff-deepu/trained_pipeline2"

# Load the pipeline
print("Loading the pipeline...")
pipeline = DDPMPipeline.from_pretrained(model_path)
print("Pipeline loaded successfully.")

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline.to(device)
print(f"Using device: {device}")

# Generate a sample image
noise = torch.randn((1, 3, 32, 32)).to(device)
print("Starting reverse diffusion process to generate an image...")

# Reverse diffusion process with progress bar
with torch.no_grad():
    for t in tqdm(reversed(range(pipeline.scheduler.config.num_train_timesteps)), desc="Generating Image"):
        noise_pred = pipeline.unet(noise, torch.tensor([t]).to(device)).sample
        noise = pipeline.scheduler.step(noise_pred, t, noise).prev_sample

        # Print some details at every 100 timesteps for analysis
        if t % 100 == 0:
            print(f"Timestep {t}: Noise prediction mean {noise_pred.mean().item():.4f}, std {noise_pred.std().item():.4f}")

# Convert to an image format and display
print("Image generation complete. Preparing for display...")
generated_image = noise.squeeze().cpu().numpy().transpose(1, 2, 0)
generated_image = (generated_image * 0.5 + 0.5).clip(0, 1)

# Display the generated image
plt.imshow(generated_image)
plt.axis("off")
plt.show()
print("Image displayed successfully.")
