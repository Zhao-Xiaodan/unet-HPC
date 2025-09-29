#!/usr/bin/env python3
"""
Dataset Size Study for Mitochondria Segmentation

This script studies how many images are sufficient for acceptable segmentation results
by training on different percentages of the full dataset (10%, 20%, 50%, 75%, 100%).

Uses the FIXED Jaccard implementation for reliable metrics.
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime
import cv2
from PIL import Image
from keras import backend, optimizers
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import random

# Get dataset percentage from environment variable (default 100%)
DATASET_PERCENTAGE = float(os.environ.get('DATASET_PERCENTAGE', '100'))

print(f"🔬 DATASET SIZE STUDY: Using {DATASET_PERCENTAGE}% of full dataset")
print("=" * 60)

# Dataset configuration
image_directory = 'dataset_full_stack/images/'
mask_directory = 'dataset_full_stack/masks/'

SIZE = 256
image_dataset = []
mask_dataset = []

# Load all available images and masks
print("📊 Loading full dataset...")
images = os.listdir(image_directory)
for i, image_name in enumerate(images):
    if (image_name.split('.')[1] == 'tif'):
        image = cv2.imread(image_directory+image_name, 1)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'tif'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))

print(f"✓ Loaded {len(image_dataset)} images and {len(mask_dataset)} masks")

# Normalize images
image_dataset = np.array(image_dataset)/255.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

# Calculate subset size based on percentage
total_samples = len(image_dataset)
subset_size = int(total_samples * DATASET_PERCENTAGE / 100)

print(f"📈 Using {subset_size} samples out of {total_samples} ({DATASET_PERCENTAGE}%)")

# Create reproducible subset using fixed random seed
np.random.seed(42)  # Fixed seed for reproducibility
indices = np.random.choice(total_samples, subset_size, replace=False)
indices = np.sort(indices)  # Sort for consistent ordering

# Extract subset
subset_images = image_dataset[indices]
subset_masks = mask_dataset[indices]

print(f"✓ Dataset subset created: {len(subset_images)} samples")

# Train/validation split
X_train, X_test, y_train, y_test = train_test_split(
    subset_images, subset_masks, test_size=0.10, random_state=42)

print(f"📋 Data split: {len(X_train)} training, {len(X_test)} validation")

# Sanity check visualization
random.seed(42)
image_number = random.randint(0, len(X_train)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256, 3)), cmap='gray')
plt.title(f'Training Image (Dataset: {DATASET_PERCENTAGE}%)')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.title(f'Training Mask (Dataset: {DATASET_PERCENTAGE}%)')
plt.savefig(f'dataset_sample_{DATASET_PERCENTAGE}pct.png', dpi=150, bbox_inches='tight')
plt.close()

# Model parameters
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
num_labels = 1  # Binary
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
batch_size = 8

# Import fixed models
from 224_225_226_models import Attention_ResUNet, UNet, Attention_UNet, dice_coef, dice_coef_loss, jacard_coef

# FOCAL LOSS
from focal_loss import BinaryFocalLoss

# Create output directory with timestamp and percentage
output_dir = f'dataset_size_study_{DATASET_PERCENTAGE}pct_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Results will be saved to: {output_dir}")

print("\n🚀 STARTING DATASET SIZE STUDY TRAINING")
print("=" * 50)
print(f"🎯 Dataset percentage: {DATASET_PERCENTAGE}%")
print(f"📊 Training samples: {len(X_train)}")
print(f"📊 Validation samples: {len(X_test)}")
print(f"🏗️  Models: UNet, Attention UNet, Attention ResUNet")
print(f"✅ Using FIXED Jaccard implementation")
print("=" * 50)

# Study results storage
study_results = {
    'dataset_percentage': DATASET_PERCENTAGE,
    'total_available_samples': total_samples,
    'used_samples': subset_size,
    'training_samples': len(X_train),
    'validation_samples': len(X_test),
    'models': {}
}

#################################################################
# UNet Training
#################################################################
print(f"\n1️⃣ TRAINING UNET ({DATASET_PERCENTAGE}% dataset)")
print("-" * 40)

unet_model = UNet(input_shape)
unet_model.compile(optimizer=Adam(lr = 1e-3), loss=BinaryFocalLoss(gamma=2),
              metrics=['accuracy', jacard_coef])

start1 = datetime.now()
unet_history = unet_model.fit(X_train, y_train,
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ),
                    shuffle=True,
                    epochs=100,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_jacard_coef',
                            patience=15,
                            restore_best_weights=True,
                            mode='max'
                        ),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='val_jacard_coef',
                            factor=0.5,
                            patience=8,
                            mode='max',
                            min_lr=1e-6
                        )
                    ])

stop1 = datetime.now()
execution_time_Unet = stop1-start1
print(f"✓ UNet training completed in: {execution_time_Unet}")

# Save UNet results
unet_model.save(f'{output_dir}/unet_{DATASET_PERCENTAGE}pct.hdf5')
best_unet_jaccard = max(unet_history.history['val_jacard_coef'])
study_results['models']['UNet'] = {
    'best_val_jaccard': best_unet_jaccard,
    'training_time': str(execution_time_Unet),
    'epochs_trained': len(unet_history.history['loss']),
    'final_val_jaccard': unet_history.history['val_jacard_coef'][-1]
}

print(f"🎯 UNet Best Validation Jaccard: {best_unet_jaccard:.4f}")

#################################################################
# Attention UNet Training
#################################################################
print(f"\n2️⃣ TRAINING ATTENTION UNET ({DATASET_PERCENTAGE}% dataset)")
print("-" * 40)

att_unet_model = Attention_UNet(input_shape)
att_unet_model.compile(optimizer=Adam(lr = 1e-3), loss=BinaryFocalLoss(gamma=2),
              metrics=['accuracy', jacard_coef])

start2 = datetime.now()
att_unet_history = att_unet_model.fit(X_train, y_train,
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ),
                    shuffle=True,
                    epochs=100,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_jacard_coef',
                            patience=15,
                            restore_best_weights=True,
                            mode='max'
                        ),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='val_jacard_coef',
                            factor=0.5,
                            patience=8,
                            mode='max',
                            min_lr=1e-6
                        )
                    ])
stop2 = datetime.now()
execution_time_Att_Unet = stop2-start2
print(f"✓ Attention UNet training completed in: {execution_time_Att_Unet}")

# Save Attention UNet results
att_unet_model.save(f'{output_dir}/attention_unet_{DATASET_PERCENTAGE}pct.hdf5')
best_att_unet_jaccard = max(att_unet_history.history['val_jacard_coef'])
study_results['models']['Attention_UNet'] = {
    'best_val_jaccard': best_att_unet_jaccard,
    'training_time': str(execution_time_Att_Unet),
    'epochs_trained': len(att_unet_history.history['loss']),
    'final_val_jaccard': att_unet_history.history['val_jacard_coef'][-1]
}

print(f"🎯 Attention UNet Best Validation Jaccard: {best_att_unet_jaccard:.4f}")

#################################################################
# Attention ResUNet Training
#################################################################
print(f"\n3️⃣ TRAINING ATTENTION RESUNET ({DATASET_PERCENTAGE}% dataset)")
print("-" * 40)

att_res_unet_model = Attention_ResUNet(input_shape)
att_res_unet_model.compile(optimizer=Adam(lr = 1e-3), loss=BinaryFocalLoss(gamma=2),
              metrics=['accuracy', jacard_coef])

start3 = datetime.now()
att_res_unet_history = att_res_unet_model.fit(X_train, y_train,
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ),
                    shuffle=True,
                    epochs=100,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_jacard_coef',
                            patience=15,
                            restore_best_weights=True,
                            mode='max'
                        ),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='val_jacard_coef',
                            factor=0.5,
                            patience=8,
                            mode='max',
                            min_lr=1e-6
                        )
                    ])

stop3 = datetime.now()
execution_time_AttResUnet = stop3-start3
print(f"✓ Attention ResUNet training completed in: {execution_time_AttResUnet}")

# Save Attention ResUNet results
att_res_unet_model.save(f'{output_dir}/attention_resunet_{DATASET_PERCENTAGE}pct.hdf5')
best_att_res_unet_jaccard = max(att_res_unet_history.history['val_jacard_coef'])
study_results['models']['Attention_ResUNet'] = {
    'best_val_jaccard': best_att_res_unet_jaccard,
    'training_time': str(execution_time_AttResUnet),
    'epochs_trained': len(att_res_unet_history.history['loss']),
    'final_val_jaccard': att_res_unet_history.history['val_jacard_coef'][-1]
}

print(f"🎯 Attention ResUNet Best Validation Jaccard: {best_att_res_unet_jaccard:.4f}")

#################################################################
# Save Training Histories
#################################################################
print(f"\n💾 SAVING TRAINING RESULTS")
print("-" * 30)

# Convert histories to DataFrames and save
unet_history_df = pd.DataFrame(unet_history.history)
att_unet_history_df = pd.DataFrame(att_unet_history.history)
att_res_unet_history_df = pd.DataFrame(att_res_unet_history.history)

unet_history_df.to_csv(f'{output_dir}/unet_history_{DATASET_PERCENTAGE}pct.csv', index=False)
att_unet_history_df.to_csv(f'{output_dir}/att_unet_history_{DATASET_PERCENTAGE}pct.csv', index=False)
att_res_unet_history_df.to_csv(f'{output_dir}/att_res_unet_history_{DATASET_PERCENTAGE}pct.csv', index=False)

# Save study results summary
with open(f'{output_dir}/study_results_{DATASET_PERCENTAGE}pct.json', 'w') as f:
    json.dump(study_results, f, indent=2)

print(f"✓ Training histories saved")
print(f"✓ Study results saved")

#################################################################
# Create Training Curves Visualization
#################################################################
print(f"\n📊 CREATING VISUALIZATIONS")
print("-" * 25)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Dataset Size Study: {DATASET_PERCENTAGE}% of Full Dataset\n'
             f'Training Samples: {len(X_train)}, Validation Samples: {len(X_test)}',
             fontsize=16, fontweight='bold')

models = ['UNet', 'Attention UNet', 'Attention ResUNet']
histories = [unet_history, att_unet_history, att_res_unet_history]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Top row: Validation Jaccard curves
for i, (model, history, color) in enumerate(zip(models, histories, colors)):
    ax = axes[0, i]
    epochs = range(1, len(history.history['val_jacard_coef']) + 1)
    ax.plot(epochs, history.history['val_jacard_coef'], color=color, linewidth=2, label='Validation')
    ax.plot(epochs, history.history['jacard_coef'], color=color, alpha=0.5, linewidth=1, label='Training')

    # Mark best epoch
    best_epoch = np.argmax(history.history['val_jacard_coef']) + 1
    best_val = max(history.history['val_jacard_coef'])
    ax.scatter(best_epoch, best_val, color=color, s=100, marker='*',
              edgecolor='black', linewidth=1, zorder=5)

    ax.set_title(f'{model}\nBest: {best_val:.3f} (epoch {best_epoch})', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Jaccard Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

# Bottom row: Loss curves
for i, (model, history, color) in enumerate(zip(models, histories, colors)):
    ax = axes[1, i]
    epochs = range(1, len(history.history['val_loss']) + 1)
    ax.plot(epochs, history.history['val_loss'], color=color, linewidth=2, label='Validation')
    ax.plot(epochs, history.history['loss'], color=color, alpha=0.5, linewidth=1, label='Training')

    ax.set_title(f'{model} - Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/training_curves_{DATASET_PERCENTAGE}pct.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Training curves saved")

#################################################################
# Final Summary
#################################################################
print(f"\n🎯 DATASET SIZE STUDY RESULTS ({DATASET_PERCENTAGE}%)")
print("=" * 60)
print(f"📊 Dataset: {subset_size} samples ({DATASET_PERCENTAGE}% of {total_samples})")
print(f"📈 Training: {len(X_train)} samples, Validation: {len(X_test)} samples")
print()
print("🏆 BEST VALIDATION JACCARD COEFFICIENTS:")
print("-" * 40)
for model_name, results in study_results['models'].items():
    print(f"{model_name:15} | {results['best_val_jaccard']:.4f} | {results['epochs_trained']:3d} epochs | {results['training_time']}")

# Calculate average performance
avg_jaccard = np.mean([results['best_val_jaccard'] for results in study_results['models'].values()])
print(f"\n📊 Average Jaccard: {avg_jaccard:.4f}")

# Determine if results are acceptable (>0.8 threshold)
acceptable_threshold = 0.8
acceptable_models = sum(1 for results in study_results['models'].values()
                       if results['best_val_jaccard'] > acceptable_threshold)
total_models = len(study_results['models'])

print(f"✅ Acceptable models (>{acceptable_threshold}): {acceptable_models}/{total_models}")

if acceptable_models == total_models:
    print(f"🎉 ALL MODELS ACHIEVE ACCEPTABLE PERFORMANCE with {DATASET_PERCENTAGE}% of dataset!")
elif acceptable_models > 0:
    print(f"⚠️  PARTIAL SUCCESS: {acceptable_models} out of {total_models} models acceptable")
else:
    print(f"❌ INSUFFICIENT DATA: No models achieve acceptable performance")

print(f"\n📁 All results saved to: {output_dir}")
print("=" * 60)
print(f"🔬 Dataset size study for {DATASET_PERCENTAGE}% completed!")