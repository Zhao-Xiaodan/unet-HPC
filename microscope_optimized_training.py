#!/usr/bin/env python3
"""
Optimized Mitochondria Segmentation Training for Microscope Dataset

Based on hyperparameter optimization results:
- Attention_UNet: LR=1e-4, Batch Size=16 (Best Performance: Val Jaccard=0.0699)
- Attention_ResUNet: LR=5e-4, Batch Size=16 (Best Performance: Val Jaccard=0.0695)
- UNet: LR=1e-3, Batch Size=8 (Best Performance: Val Jaccard=0.0670)

All models use:
- Gradient clipping (clipnorm=1.0) for stability
- Binary Focal Loss (gamma=2)
- Extended early stopping (patience=15)
- Adaptive learning rate reduction
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from datetime import datetime
import cv2
from PIL import Image
from keras import backend

# Dataset paths - UPDATE TO MICROSCOPE DATASET
image_directory = 'dataset_microscope/images/'
mask_directory = 'dataset_microscope/masks/'

# Image parameters
SIZE = 256
image_dataset = []
mask_dataset = []

print("=" * 70)
print("OPTIMIZED MITOCHONDRIA SEGMENTATION - MICROSCOPE DATASET")
print("=" * 70)
print(f"Dataset: {image_directory}")
print(f"Image size: {SIZE}x{SIZE}")
print("")

# Load and preprocess images
print("Loading images...")
images = os.listdir(image_directory)
for i, image_name in enumerate(images):
    if (image_name.split('.')[-1] in ['tif', 'tiff', 'png', 'jpg']):
        image = cv2.imread(image_directory + image_name, 1)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

print(f"Loaded {len(image_dataset)} images")

# Load and preprocess masks
print("Loading masks...")
masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[-1] in ['tif', 'tiff', 'png', 'jpg']):
        image = cv2.imread(mask_directory + image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))

print(f"Loaded {len(mask_dataset)} masks")

# Normalize images
image_dataset = np.array(image_dataset) / 255.
# Rescale masks to 0-1
mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

print(f"Image dataset shape: {image_dataset.shape}")
print(f"Mask dataset shape: {mask_dataset.shape}")

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    image_dataset, mask_dataset, test_size=0.10, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print("")

# Sanity check visualization
import random
image_number = random.randint(0, len(X_train) - 1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256, 3)), cmap='gray')
plt.title('Training Image Sample')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.title('Training Mask Sample')
plt.savefig('dataset_sample_check.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved dataset sample visualization: dataset_sample_check.png")
print("")

# Model parameters
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
num_labels = 1  # Binary segmentation
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

print("=" * 70)
print("MODEL CONFIGURATION")
print("=" * 70)
print(f"Input shape: {input_shape}")
print("")

# Import focal loss
from focal_loss import BinaryFocalLoss

# Import model architectures
from models import Attention_ResUNet, UNet, Attention_UNet, jacard_coef

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"microscope_training_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")
print("")

# Training results storage
training_results = []

###############################################################################
# MODEL 1: STANDARD U-NET
# Optimized hyperparameters: LR=1e-3, Batch Size=8
###############################################################################

print("=" * 70)
print("TRAINING MODEL 1/3: STANDARD U-NET")
print("=" * 70)
print("Optimized Hyperparameters:")
print("  Learning Rate: 1e-3")
print("  Batch Size: 8")
print("  Expected Val Jaccard: ~0.0670")
print("")

unet_model = UNet(input_shape)
unet_model.compile(
    optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),  # Gradient clipping for stability
    loss=BinaryFocalLoss(gamma=2),
    metrics=['accuracy', jacard_coef]
)

callbacks_unet = [
    EarlyStopping(
        monitor='val_jacard_coef',
        mode='max',
        patience=15,  # Extended patience
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        f'{output_dir}/best_unet_model.hdf5',
        monitor='val_jacard_coef',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

start_unet = datetime.now()
unet_history = unet_model.fit(
    X_train, y_train,
    verbose=1,
    batch_size=8,  # Optimized batch size
    validation_data=(X_test, y_test),
    shuffle=True,
    epochs=100,
    callbacks=callbacks_unet
)
stop_unet = datetime.now()

execution_time_unet = stop_unet - start_unet
print(f"✓ UNet training completed in {execution_time_unet}")

# Save model and history
unet_model.save(f'{output_dir}/final_unet_model.hdf5')

import pandas as pd
unet_history_df = pd.DataFrame(unet_history.history)
unet_history_df.to_csv(f'{output_dir}/unet_history.csv', index=False)

best_val_jacard_unet = unet_history_df['val_jacard_coef'].max()
training_results.append({
    'model': 'UNet',
    'lr': 1e-3,
    'batch_size': 8,
    'best_val_jacard': best_val_jacard_unet,
    'training_time': str(execution_time_unet)
})

print(f"Best Val Jaccard: {best_val_jacard_unet:.4f}")
print("")

###############################################################################
# MODEL 2: ATTENTION U-NET
# Optimized hyperparameters: LR=1e-4, Batch Size=16 (BEST OVERALL)
###############################################################################

print("=" * 70)
print("TRAINING MODEL 2/3: ATTENTION U-NET (BEST PERFORMER)")
print("=" * 70)
print("Optimized Hyperparameters:")
print("  Learning Rate: 1e-4")
print("  Batch Size: 16")
print("  Expected Val Jaccard: ~0.0699 (highest)")
print("")

att_unet_model = Attention_UNet(input_shape)
att_unet_model.compile(
    optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),  # Lower LR for attention model
    loss=BinaryFocalLoss(gamma=2),
    metrics=['accuracy', jacard_coef]
)

callbacks_att_unet = [
    EarlyStopping(
        monitor='val_jacard_coef',
        mode='max',
        patience=15,
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        f'{output_dir}/best_attention_unet_model.hdf5',
        monitor='val_jacard_coef',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

start_att_unet = datetime.now()
att_unet_history = att_unet_model.fit(
    X_train, y_train,
    verbose=1,
    batch_size=16,  # Optimized batch size
    validation_data=(X_test, y_test),
    shuffle=True,
    epochs=100,
    callbacks=callbacks_att_unet
)
stop_att_unet = datetime.now()

execution_time_att_unet = stop_att_unet - start_att_unet
print(f"✓ Attention UNet training completed in {execution_time_att_unet}")

# Save model and history
att_unet_model.save(f'{output_dir}/final_attention_unet_model.hdf5')

att_unet_history_df = pd.DataFrame(att_unet_history.history)
att_unet_history_df.to_csv(f'{output_dir}/attention_unet_history.csv', index=False)

best_val_jacard_att_unet = att_unet_history_df['val_jacard_coef'].max()
training_results.append({
    'model': 'Attention_UNet',
    'lr': 1e-4,
    'batch_size': 16,
    'best_val_jacard': best_val_jacard_att_unet,
    'training_time': str(execution_time_att_unet)
})

print(f"Best Val Jaccard: {best_val_jacard_att_unet:.4f}")
print("")

###############################################################################
# MODEL 3: ATTENTION RESIDUAL U-NET
# Optimized hyperparameters: LR=5e-4, Batch Size=16
###############################################################################

print("=" * 70)
print("TRAINING MODEL 3/3: ATTENTION RESIDUAL U-NET")
print("=" * 70)
print("Optimized Hyperparameters:")
print("  Learning Rate: 5e-4")
print("  Batch Size: 16")
print("  Expected Val Jaccard: ~0.0695")
print("")

att_res_unet_model = Attention_ResUNet(input_shape)
att_res_unet_model.compile(
    optimizer=Adam(learning_rate=5e-4, clipnorm=1.0),  # Medium LR for residual attention
    loss=BinaryFocalLoss(gamma=2),
    metrics=['accuracy', jacard_coef]
)

callbacks_att_res_unet = [
    EarlyStopping(
        monitor='val_jacard_coef',
        mode='max',
        patience=15,
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        f'{output_dir}/best_attention_resunet_model.hdf5',
        monitor='val_jacard_coef',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

start_att_res_unet = datetime.now()
att_res_unet_history = att_res_unet_model.fit(
    X_train, y_train,
    verbose=1,
    batch_size=16,  # Optimized batch size
    validation_data=(X_test, y_test),
    shuffle=True,
    epochs=100,
    callbacks=callbacks_att_res_unet
)
stop_att_res_unet = datetime.now()

execution_time_att_res_unet = stop_att_res_unet - start_att_res_unet
print(f"✓ Attention ResUNet training completed in {execution_time_att_res_unet}")

# Save model and history
att_res_unet_model.save(f'{output_dir}/final_attention_resunet_model.hdf5')

att_res_unet_history_df = pd.DataFrame(att_res_unet_history.history)
att_res_unet_history_df.to_csv(f'{output_dir}/attention_resunet_history.csv', index=False)

best_val_jacard_att_res_unet = att_res_unet_history_df['val_jacard_coef'].max()
training_results.append({
    'model': 'Attention_ResUNet',
    'lr': 5e-4,
    'batch_size': 16,
    'best_val_jacard': best_val_jacard_att_res_unet,
    'training_time': str(execution_time_att_res_unet)
})

print(f"Best Val Jaccard: {best_val_jacard_att_res_unet:.4f}")
print("")

###############################################################################
# GENERATE COMPARISON VISUALIZATIONS
###############################################################################

print("=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Training curves comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# UNet plots
axes[0, 0].plot(unet_history_df['loss'], label='Train Loss', color='blue')
axes[0, 0].plot(unet_history_df['val_loss'], label='Val Loss', color='red')
axes[0, 0].set_title('UNet - Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(unet_history_df['jacard_coef'], label='Train Jaccard', color='blue')
axes[1, 0].plot(unet_history_df['val_jacard_coef'], label='Val Jaccard', color='red')
axes[1, 0].set_title('UNet - Jaccard Coefficient')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Jaccard')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Attention UNet plots
axes[0, 1].plot(att_unet_history_df['loss'], label='Train Loss', color='blue')
axes[0, 1].plot(att_unet_history_df['val_loss'], label='Val Loss', color='red')
axes[0, 1].set_title('Attention UNet - Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(att_unet_history_df['jacard_coef'], label='Train Jaccard', color='blue')
axes[1, 1].plot(att_unet_history_df['val_jacard_coef'], label='Val Jaccard', color='red')
axes[1, 1].set_title('Attention UNet - Jaccard Coefficient')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Jaccard')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Attention ResUNet plots
axes[0, 2].plot(att_res_unet_history_df['loss'], label='Train Loss', color='blue')
axes[0, 2].plot(att_res_unet_history_df['val_loss'], label='Val Loss', color='red')
axes[0, 2].set_title('Attention ResUNet - Loss')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

axes[1, 2].plot(att_res_unet_history_df['jacard_coef'], label='Train Jaccard', color='blue')
axes[1, 2].plot(att_res_unet_history_df['val_jacard_coef'], label='Val Jaccard', color='red')
axes[1, 2].set_title('Attention ResUNet - Jaccard Coefficient')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Jaccard')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/training_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved training curves: {output_dir}/training_curves_comparison.png")

# Performance summary bar chart
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
models = [r['model'] for r in training_results]
jaccards = [r['best_val_jacard'] for r in training_results]
lrs = [r['lr'] for r in training_results]
batch_sizes = [r['batch_size'] for r in training_results]

bars = ax.bar(models, jaccards, color=['#3498db', '#e74c3c', '#2ecc71'])
ax.set_ylabel('Best Validation Jaccard Coefficient', fontsize=12)
ax.set_title('Model Performance Comparison - Microscope Dataset', fontsize=14, fontweight='bold')
ax.set_ylim([0, max(jaccards) * 1.1])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, jacard, lr, bs) in enumerate(zip(bars, jaccards, lrs, batch_sizes)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{jacard:.4f}\nLR={lr:.0e}\nBS={bs}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/performance_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved performance summary: {output_dir}/performance_summary.png")

###############################################################################
# FINAL SUMMARY REPORT
###############################################################################

print("")
print("=" * 70)
print("TRAINING COMPLETE - SUMMARY REPORT")
print("=" * 70)
print("")

results_df = pd.DataFrame(training_results)
print(results_df.to_string(index=False))
print("")

print("Output Files:")
print(f"  Directory: {output_dir}/")
print(f"  - Model files: best_*.hdf5, final_*.hdf5")
print(f"  - Training histories: *_history.csv")
print(f"  - Visualizations: training_curves_comparison.png, performance_summary.png")
print("")

# Determine best model
best_model_idx = results_df['best_val_jacard'].idxmax()
best_model = results_df.iloc[best_model_idx]

print("🏆 BEST MODEL:")
print(f"  Architecture: {best_model['model']}")
print(f"  Learning Rate: {best_model['lr']:.0e}")
print(f"  Batch Size: {best_model['batch_size']}")
print(f"  Best Val Jaccard: {best_model['best_val_jacard']:.4f}")
print(f"  Training Time: {best_model['training_time']}")
print("")

print("=" * 70)
print("All training completed successfully!")
print("=" * 70)
