Guide: Visualizing U-Net Intermediate Features with PyTorch
This document provides a complete guide for creating a Python script to visualize the internal workings of your trained U-Net models. It is based directly on the model definitions and preprocessing functions in your train_pytorch_comparison_no_aug.py file.
The guide covers two distinct visualization techniques:
"What the feature map looks like": Directly plotting the activation (output) tensors from any layer.
"What the original input looks like": Reconstructing the input image from a layer's activation, also known as feature inversion.
Part 0: Script Setup
Your final visualization script will need to perform these setup steps.
1. Imports
Import all necessary libraries:
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# === CRITICAL: Import from your training script ===
# You must copy/paste the following classes and functions
# directly from 'train_pytorch_comparison_no_aug.py'
# into your new visualization script.

# 1. Preprocessing Function
def _percentile_norm(arr: np.ndarray):
    # (Copy the full function from your script)
    lo, hi = np.percentile(arr, [0.5, 99.5]).astype(np.float32)
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr.astype(np.float32)

# 2. Model Architecture Definitions
# (Copy all of them: ConvBlock, ResConvBlock, AttentionGate,
#  UNet, AttentionUNet, AttentionResUNet)

class ConvBlock(nn.Module):
    # (Copy the full class)
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class UNet(nn.Module):
    # (Copy the full UNet class definition)
    # ... (include enc1, pool1, enc2, pool2, ..., dec1, out)
    def __init__(self, in_channels=1, n_filters=32, dropout=0.1):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, n_filters, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(n_filters, n_filters * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(n_filters * 2, n_filters * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(n_filters * 4, n_filters * 8, dropout)
        self.pool4 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = ConvBlock(n_filters * 8, n_filters * 16, dropout)
        # Decoder
        self.up4 = nn.ConvTranspose2d(n_filters * 16, n_filters * 8, 2, stride=2)
        self.dec4 = ConvBlock(n_filters * 16, n_filters * 8, dropout)
        self.up3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 2, stride=2)
        self.dec3 = ConvBlock(n_filters * 8, n_filters * 4, dropout)
        self.up2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 2, stride=2)
        self.dec2 = ConvBlock(n_filters * 4, n_filters * 2, dropout)
        self.up1 = nn.ConvTranspose2d(n_filters * 2, n_filters, 2, stride=2)
        self.dec1 = ConvBlock(n_filters * 2, n_filters, dropout)
        # Output
        self.out = nn.Conv2d(n_filters, 1, 1)

    def forward(self, x):
        # (Copy the full forward method)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)

# (Also copy AttentionUNet, ResConvBlock, etc. if you
#  plan to visualize those models)


2. Helper Functions
Add these helper functions for preprocessing and postprocessing.
def preprocess_image(img_path, img_size=512):
    """
    Loads and preprocesses a single image, matching
    your BeadsDataset implementation.
    """
    # Load as grayscale
    im = Image.open(str(img_path)).convert("L")
    arr = np.array(im, dtype=np.float32)

    # Apply percentile normalization
    arr_norm = _percentile_norm(arr)

    # Convert to tensor
    tensor = torch.from_numpy(arr_norm).unsqueeze(0)  # [1, H, W]

    # Resize
    if tensor.shape[1] != img_size or tensor.shape[2] != img_size:
        tensor = F.interpolate(tensor.unsqueeze(0),
                             size=(img_size, img_size),
                             mode="bilinear",
                             align_corners=False).squeeze(0)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)  # [1, 1, H, W]
    return tensor

def postprocess_tensor(tensor):
    """
    Converts a tensor from [0, 1] range back to a
    plottable numpy image.
    """
    image = tensor.detach().cpu().squeeze().numpy()
    image = np.clip(image, 0, 1)
    return image


3. Load Your Trained Model
Define the parameters for the specific model you trained and want to visualize.
# --- Configuration ---
# (Adjust these to match your trained model)
MODEL_ARCH = 'unet' # 'unet', 'attention_unet', etc.
N_FILTERS = 32
DROPOUT = 0.1
MODEL_PATH = './pytorch_comparison_no_aug_20251027_160000/unet/checkpoints/unet_n_filters32_dropout0.1_learning_rate0.003/best_model.pth'
IMAGE_PATH = './dataset_shrunk_masks/images/your_test_image.png'
IMG_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load Model ---
print(f"Loading model: {MODEL_ARCH}")
if MODEL_ARCH == 'unet':
    model = UNet(in_channels=1, n_filters=N_FILTERS, dropout=DROPOUT)
# elif MODEL_ARCH == 'attention_unet':
#    model = AttentionUNet(...)
# ... (add other archs as needed)

# Load the weights
# Note: your checkpoint saves more than just the state_dict
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval() # CRITICAL: Set model to evaluation mode
print("Model loaded successfully.")


Part 1: "What the feature map looks like" (Direct Visualization)
This technique uses PyTorch Hooks to "hook" into a layer and copy its output during a forward pass.
1. Define the Hook
The hook captures the output of a layer and stores it in a dictionary.
# A dictionary to store the captured activations
activations = {}

def get_activation(name):
    """
    Returns a hook function that saves the output of a
    layer to the 'activations' dictionary.
    """
    def hook(model, input, output):
        # Detach the tensor to save memory
        activations[name] = output.detach()
    return hook


2. Register Hooks on Desired Layers
You must use the exact variable names defined in your UNet class.
Valid Layer Names in UNet:
model.enc1 (output of 1st conv block)
model.enc2
model.enc3
model.enc4
model.bottleneck
model.dec4 (output of 1st decoder conv block)
model.dec3
model.dec2
model.dec1 (output of final decoder conv block)
model.out (the final 1x1 conv, pre-sigmoid)
# --- Register Hooks ---
# Choose which layers to visualize
layers_to_visualize = {
    'encoder_1': model.enc1,
    'bottleneck': model.bottleneck,
    'decoder_1': model.dec1
}

hooks = {}
for name, layer in layers_to_visualize.items():
    hooks[name] = layer.register_forward_hook(get_activation(name))
    print(f"Registered hook for: {name}")


3. Run the Model & Plot Activations
A single forward pass will trigger all the hooks and populate the activations dictionary.
# --- Load and Preprocess Image ---
image_tensor = preprocess_image(IMAGE_PATH, IMG_SIZE).to(DEVICE)

# --- Run Forward Pass ---
# This will trigger the hooks
with torch.no_grad():
    _ = model(image_tensor)

print("\n--- Activations Captured ---")

# --- Plotting Function ---
def plot_feature_maps(maps, layer_name, max_cols=8):
    maps = maps.cpu().squeeze(0) # Remove batch dim
    num_maps = maps.shape[0]

    # Determine grid size
    max_cols = min(num_maps, max_cols)
    num_rows = int(np.ceil(num_maps / max_cols))

    fig, axes = plt.subplots(num_rows, max_cols,
                             figsize=(max_cols * 2, num_rows * 2))
    fig.suptitle(f"Feature Maps: {layer_name} (Shape: {list(maps.shape)})", fontsize=16)

    # Flatten axes array for easy iteration
    if num_rows > 1:
        axes = axes.flatten()
    elif num_maps == 1:
        axes = [axes]

    for i in range(num_maps):
        ax = axes[i]
        ax.imshow(maps[i], cmap='viridis') # 'viridis' or 'gray'
        ax.set_title(f"Channel {i}")
        ax.axis('off')

    # Turn off unused subplots
    for j in range(num_maps, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Plot the captured activations ---
for name, activation in activations.items():
    plot_feature_maps(activation, name)

# --- Remove hooks (important for memory) ---
for handle in hooks.values():
    handle.remove()


Part 2: "What the original input looks like" (Feature Inversion)
This technique starts with random noise and optimizes it until its feature map at a specific layer matches the target feature map from your real image.
1. Helper Function: Total Variation (TV) Loss
This is a regularizer that encourages the generated image to be smoother and more "natural."
def total_variation_loss(img):
    """
    Computes the Total Variation Loss for a batch of images.
    Helps to reduce noise in the generated image.
    """
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (img.shape[0] * img.shape[1] * img.shape[2] * img.shape[3])


2. Feature Inversion Function
This function contains the full optimization loop.
def reconstruct_from_layer(model, target_image_tensor, layer_hook_name, layer_module,
                           n_steps=2000, lr=0.1, tv_weight=0.001):

    print(f"\n--- Starting Feature Inversion for: {layer_hook_name} ---")

    # --- Step 1: Get Target Feature Map ---
    # We re-use the hook logic from Part 1
    target_activations = {}

    def get_target_hook(name):
        def hook(model, input, output):
            target_activations[name] = output.detach()
        return hook

    hook_handle = layer_module.register_forward_hook(get_target_hook(layer_hook_name))

    # Run forward pass on the REAL image
    with torch.no_grad():
        _ = model(target_image_tensor)

    # This is the feature map we want to match
    target_feature_map = target_activations[layer_hook_name].clone()
    hook_handle.remove() # Remove this hook

    print(f"Target feature map shape: {target_feature_map.shape}")

    # --- Step 2: Create Image to Optimize ---
    # Start with random noise
    optimized_image = torch.randn(
        target_image_tensor.shape,
        device=DEVICE,
        requires_grad=True
    )

    # --- Step 3: Setup Optimizer ---
    # The optimizer will ONLY update the pixels of optimized_image
    optimizer = torch.optim.Adam([optimized_image], lr=lr)

    # We will re-use the *same* hook logic, but this time
    # it will capture the features from the 'optimized_image'
    current_activations = {}

    def get_current_hook(name):
        def hook(model, input, output):
            current_activations[name] = output # No detach, keep in graph
        return hook

    hook_handle = layer_module.register_forward_hook(get_current_hook(layer_hook_name))

    # Loss function for the features
    feature_loss_fn = nn.MSELoss()

    # --- Step 4: The Optimization Loop ---
    for i in range(n_steps + 1):
        optimizer.zero_grad()

        # Pass the NOISE image through the model
        _ = model(optimized_image)

        # Get the feature map for the noise image
        current_feature_map = current_activations[layer_hook_name]

        # Calculate loss
        feature_loss = feature_loss_fn(current_feature_map, target_feature_map)
        tv_loss = total_variation_loss(optimized_image)

        # Combine losses
        total_loss = feature_loss + (tv_weight * tv_loss)

        # Backpropagate
        total_loss.backward()
        optimizer.step()

        # Clamp image values to [0, 1] range after each step
        with torch.no_grad():
            optimized_image.data.clamp_(0, 1)

        if i % 200 == 0:
            print(f"Step {i:4d}/{n_steps} | Total Loss: {total_loss.item():.4f} | "
                  f"Feature Loss: {feature_loss.item():.4f} | TV Loss: {tv_loss.item():.4f}")

    hook_handle.remove()
    print("Optimization finished.")

    return optimized_image, target_feature_map


3. Run the Feature Inversion
Now, call the function for the layers you want to invert.
# --- Load the real image (we already did this) ---
# image_tensor = preprocess_image(IMAGE_PATH, IMG_SIZE).to(DEVICE)

# --- Choose a layer to reconstruct from ---
# Note: Deeper layers (like bottleneck) are more interesting
# but harder to reconstruct.
layer_to_invert_name = 'enc4'
layer_to_invert_module = model.enc4

# --- Run the inversion ---
reconstructed_image, _ = reconstruct_from_layer(
    model,
    image_tensor,
    layer_to_invert_name,
    layer_to_invert_module,
    n_steps=2000,
    lr=0.01,         # Use a smaller LR for inversion
    tv_weight=0.001  # Adjust this to make image smoother/noisier
)

# --- Plot the results ---
# Post-process for plotting
original_img_plot = postprocess_tensor(image_tensor)
recon_img_plot = postprocess_tensor(reconstructed_image)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(f"Feature Inversion from Layer: '{layer_to_invert_name}'", fontsize=16)

ax1.imshow(original_img_plot, cmap='gray')
ax1.set_title("Original Image")
ax1.axis('off')

ax2.imshow(recon_img_plot, cmap='gray')
ax2.set_title("Reconstructed Image")
ax2.axis('off')

plt.show()

# You can now try again with a different layer, e.g.:
# layer_to_invert_name = 'bottleneck'
# layer_to_invert_module = model.bottleneck
# ... and re-run the inversion



