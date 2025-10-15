#!/usr/bin/env python3
"""
Mitochondria Segmentation Training - Xukuang Parameters on Shrunk Masks Dataset
================================================================================

Trains 3 U-Net architectures using Xukuang's parameters from bead_seg.ipynb:
- Standard U-Net
- Attention U-Net
- Attention Residual U-Net

Dataset: dataset_shrunk_masks/
Image Size: 512×512
Loss: Binary Focal Loss (gamma=2)
Learning Rate: 5e-3
Epochs: 200
Batch Size: 4

Based on: bead_seg.ipynb
Author: Dr. Sreenivas Bhattiprolu (modified for shrunk masks dataset)
Date: October 15, 2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime
import cv2
from PIL import Image
from tensorflow.keras import backend
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from pathlib import Path

# Import custom modules
from models import Attention_ResUNet, UNet, Attention_UNet, dice_coef, dice_coef_loss, jacard_coef
from focal_loss import BinaryFocalLoss

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

print("=" * 70)
print("GPU CONFIGURATION")
print("=" * 70)

gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {len(gpus)}")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"Memory growth setting error: {e}")

print(f"TensorFlow version: {tf.__version__}")
print("=" * 70)
print()

# ============================================================================
# XUKUANG'S PARAMETERS (from bead_seg.ipynb)
# ============================================================================

# Training parameters
LEARNING_RATE = 5e-3  # lr = 5e-3
EPOCHS = 200          # epochs = 200
BATCH_SIZE = 4        # batch_size = 4
SIZE = 512            # SIZE = 512
TEST_SIZE = 0.2       # test_size = 0.2
RANDOM_STATE = 0      # random_state = 0

# Loss function: BinaryFocalLoss(gamma=2)
FOCAL_GAMMA = 2

# Dataset paths
IMAGE_DIRECTORY = './dataset_shrunk_masks/images/'
MASK_DIRECTORY = './dataset_shrunk_masks/masks/'

# Output directory with timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"./xukuang_params_shrunk_{TIMESTAMP}")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("XUKUANG'S TRAINING PARAMETERS")
print("=" * 70)
print(f"Learning Rate:     {LEARNING_RATE}")
print(f"Epochs:            {EPOCHS}")
print(f"Batch Size:        {BATCH_SIZE}")
print(f"Image Size:        {SIZE}×{SIZE}")
print(f"Loss Function:     BinaryFocalLoss(gamma={FOCAL_GAMMA})")
print(f"Test Split:        {TEST_SIZE * 100}%")
print(f"Random State:      {RANDOM_STATE}")
print(f"Dataset:           {IMAGE_DIRECTORY}")
print(f"Output Directory:  {OUTPUT_DIR}")
print("=" * 70)
print()

# ============================================================================
# LOAD AND PREPROCESS DATASET
# ============================================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

image_dataset = []
mask_dataset = []

# Load images
images = sorted(os.listdir(IMAGE_DIRECTORY))
print(f"Found {len(images)} files in image directory")

for i, image_name in enumerate(images):
    if image_name.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'tif', 'tiff']:
        image = cv2.imread(os.path.join(IMAGE_DIRECTORY, image_name), 1)
        if image is None:
            print(f"Warning: Could not read {image_name}")
            continue
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

print(f"Loaded {len(image_dataset)} images")

# Load masks
masks = sorted(os.listdir(MASK_DIRECTORY))
print(f"Found {len(masks)} files in mask directory")

for i, image_name in enumerate(masks):
    if image_name.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'tif', 'tiff']:
        image = cv2.imread(os.path.join(MASK_DIRECTORY, image_name), 0)
        if image is None:
            print(f"Warning: Could not read {image_name}")
            continue
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))

print(f"Loaded {len(mask_dataset)} masks")

# Normalize
image_dataset = np.array(image_dataset) / 255.
mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

print(f"Image dataset shape: {image_dataset.shape}")
print(f"Mask dataset shape: {mask_dataset.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    image_dataset, mask_dataset,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print("=" * 70)
print()

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
num_labels = 1  # Binary segmentation
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

print("=" * 70)
print("MODEL CONFIGURATION")
print("=" * 70)
print(f"Input Shape: {input_shape}")
print(f"Image Height: {IMG_HEIGHT}")
print(f"Image Width: {IMG_WIDTH}")
print(f"Image Channels: {IMG_CHANNELS}")
print(f"Number of Labels: {num_labels}")
print("=" * 70)
print()

# ============================================================================
# TRAINING METADATA
# ============================================================================

experiment_info = {
    'timestamp': TIMESTAMP,
    'dataset': 'dataset_shrunk_masks',
    'training_parameters': {
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'image_size': SIZE,
        'loss_function': f'BinaryFocalLoss(gamma={FOCAL_GAMMA})',
        'optimizer': 'Adam',
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
    },
    'dataset_info': {
        'total_images': len(image_dataset),
        'training_samples': int(X_train.shape[0]),
        'testing_samples': int(X_test.shape[0]),
        'image_directory': IMAGE_DIRECTORY,
        'mask_directory': MASK_DIRECTORY,
    },
    'model_info': {
        'input_shape': input_shape,
        'architectures': ['UNet', 'Attention_UNet', 'Attention_ResUNet'],
    },
    'source': 'bead_seg.ipynb (Xukuang parameters)',
}

# Save metadata
with open(OUTPUT_DIR / 'EXPERIMENT_INFO.json', 'w') as f:
    json.dump(experiment_info, f, indent=4)

print(f"✓ Saved experiment metadata to {OUTPUT_DIR / 'EXPERIMENT_INFO.json'}")
print()

# ============================================================================
# TRAIN MODEL 1: U-Net
# ============================================================================

print("=" * 70)
print("TRAINING MODEL 1: U-NET")
print("=" * 70)

unet_model = UNet(input_shape)
unet_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=BinaryFocalLoss(gamma=FOCAL_GAMMA),
    metrics=['accuracy', jacard_coef]
)

print(f"Total parameters: {unet_model.count_params():,}")
print()

start1 = datetime.now()
print(f"Training started: {start1}")

unet_history = unet_model.fit(
    X_train, y_train,
    verbose=1,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    shuffle=False,
    epochs=EPOCHS
)

stop1 = datetime.now()
execution_time_Unet = stop1 - start1
print(f"Training completed: {stop1}")
print(f"UNet execution time: {execution_time_Unet}")

# Save model
unet_model_path = OUTPUT_DIR / 'unet_xukuang_params_shrunk.keras'
unet_model.save(unet_model_path)
print(f"✓ Saved model: {unet_model_path}")

# Save history
unet_history_df = pd.DataFrame(unet_history.history)
unet_history_csv = OUTPUT_DIR / 'unet_history.csv'
unet_history_df.to_csv(unet_history_csv, index=False)
print(f"✓ Saved history: {unet_history_csv}")
print("=" * 70)
print()

# ============================================================================
# TRAIN MODEL 2: Attention U-Net
# ============================================================================

print("=" * 70)
print("TRAINING MODEL 2: ATTENTION U-NET")
print("=" * 70)

att_unet_model = Attention_UNet(input_shape)
att_unet_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=BinaryFocalLoss(gamma=FOCAL_GAMMA),
    metrics=['accuracy', jacard_coef]
)

print(f"Total parameters: {att_unet_model.count_params():,}")
print()

start2 = datetime.now()
print(f"Training started: {start2}")

att_unet_history = att_unet_model.fit(
    X_train, y_train,
    verbose=1,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    shuffle=False,
    epochs=EPOCHS
)

stop2 = datetime.now()
execution_time_Att_Unet = stop2 - start2
print(f"Training completed: {stop2}")
print(f"Attention UNet execution time: {execution_time_Att_Unet}")

# Save model
att_unet_model_path = OUTPUT_DIR / 'attention_unet_xukuang_params_shrunk.keras'
att_unet_model.save(att_unet_model_path)
print(f"✓ Saved model: {att_unet_model_path}")

# Save history
att_unet_history_df = pd.DataFrame(att_unet_history.history)
att_unet_history_csv = OUTPUT_DIR / 'attention_unet_history.csv'
att_unet_history_df.to_csv(att_unet_history_csv, index=False)
print(f"✓ Saved history: {att_unet_history_csv}")
print("=" * 70)
print()

# ============================================================================
# TRAIN MODEL 3: Attention Residual U-Net
# ============================================================================

print("=" * 70)
print("TRAINING MODEL 3: ATTENTION RESIDUAL U-NET")
print("=" * 70)

att_res_unet_model = Attention_ResUNet(input_shape)
att_res_unet_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=BinaryFocalLoss(gamma=FOCAL_GAMMA),
    metrics=['accuracy', jacard_coef]
)

print(f"Total parameters: {att_res_unet_model.count_params():,}")
print()

start3 = datetime.now()
print(f"Training started: {start3}")

att_res_unet_history = att_res_unet_model.fit(
    X_train, y_train,
    verbose=1,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    shuffle=False,
    epochs=EPOCHS
)

stop3 = datetime.now()
execution_time_AttResUnet = stop3 - start3
print(f"Training completed: {stop3}")
print(f"Attention ResUNet execution time: {execution_time_AttResUnet}")

# Save model
att_res_unet_model_path = OUTPUT_DIR / 'attention_resunet_xukuang_params_shrunk.keras'
att_res_unet_model.save(att_res_unet_model_path)
print(f"✓ Saved model: {att_res_unet_model_path}")

# Save history
att_res_unet_history_df = pd.DataFrame(att_res_unet_history.history)
att_res_unet_history_csv = OUTPUT_DIR / 'attention_resunet_history.csv'
att_res_unet_history_df.to_csv(att_res_unet_history_csv, index=False)
print(f"✓ Saved history: {att_res_unet_history_csv}")
print("=" * 70)
print()

# ============================================================================
# TRAINING SUMMARY
# ============================================================================

print("=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"Output Directory: {OUTPUT_DIR}")
print()
print("Execution Times:")
print(f"  UNet:                  {execution_time_Unet}")
print(f"  Attention UNet:        {execution_time_Att_Unet}")
print(f"  Attention ResUNet:     {execution_time_AttResUnet}")
print(f"  Total:                 {execution_time_Unet + execution_time_Att_Unet + execution_time_AttResUnet}")
print()
print("Final Validation Metrics:")
print(f"  UNet:")
print(f"    Loss:        {unet_history.history['val_loss'][-1]:.4f}")
print(f"    Accuracy:    {unet_history.history['val_accuracy'][-1]:.4f}")
print(f"    Jaccard:     {unet_history.history['val_jacard_coef'][-1]:.4f}")
print(f"  Attention UNet:")
print(f"    Loss:        {att_unet_history.history['val_loss'][-1]:.4f}")
print(f"    Accuracy:    {att_unet_history.history['val_accuracy'][-1]:.4f}")
print(f"    Jaccard:     {att_unet_history.history['val_jacard_coef'][-1]:.4f}")
print(f"  Attention ResUNet:")
print(f"    Loss:        {att_res_unet_history.history['val_loss'][-1]:.4f}")
print(f"    Accuracy:    {att_res_unet_history.history['val_accuracy'][-1]:.4f}")
print(f"    Jaccard:     {att_res_unet_history.history['val_jacard_coef'][-1]:.4f}")
print("=" * 70)

# Save summary
summary = {
    'execution_times': {
        'unet': str(execution_time_Unet),
        'attention_unet': str(execution_time_Att_Unet),
        'attention_resunet': str(execution_time_AttResUnet),
        'total': str(execution_time_Unet + execution_time_Att_Unet + execution_time_AttResUnet),
    },
    'final_validation_metrics': {
        'unet': {
            'loss': float(unet_history.history['val_loss'][-1]),
            'accuracy': float(unet_history.history['val_accuracy'][-1]),
            'jaccard': float(unet_history.history['val_jacard_coef'][-1]),
        },
        'attention_unet': {
            'loss': float(att_unet_history.history['val_loss'][-1]),
            'accuracy': float(att_unet_history.history['val_accuracy'][-1]),
            'jaccard': float(att_unet_history.history['val_jacard_coef'][-1]),
        },
        'attention_resunet': {
            'loss': float(att_res_unet_history.history['val_loss'][-1]),
            'accuracy': float(att_res_unet_history.history['val_accuracy'][-1]),
            'jaccard': float(att_res_unet_history.history['val_jacard_coef'][-1]),
        },
    },
}

with open(OUTPUT_DIR / 'TRAINING_SUMMARY.json', 'w') as f:
    json.dump(summary, f, indent=4)

print(f"\n✓ Saved training summary to {OUTPUT_DIR / 'TRAINING_SUMMARY.json'}")
print("\n🎉 TRAINING COMPLETE!")
