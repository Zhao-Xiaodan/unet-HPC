#!/usr/bin/env python3
"""
Optimized Microbead Segmentation Training

Key differences from mitochondria training:
- Dice Loss instead of Focal Loss (for balanced classes)
- Lower learning rate (1e-4 vs 1e-3) for dense objects
- Larger batch size (32 vs 8-16) for gradient stability
- Dropout regularization (0.3) to prevent overfitting
- Data augmentation for uniform circular objects
- Stratified train/val split by object density

Based on dataset analysis recommendations
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras import backend as K
from datetime import datetime
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

print("=" * 80)
print("OPTIMIZED MICROBEAD SEGMENTATION TRAINING")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset paths
image_directory = 'dataset_microscope/images/'
mask_directory = 'dataset_microscope/masks/'

# Image parameters
SIZE = 256

# OPTIMIZED HYPERPARAMETERS FOR MICROBEADS
BATCH_SIZE = 32          # Larger than mitochondria (8-16)
LEARNING_RATE = 1e-4     # Lower than mitochondria (1e-3)
DROPOUT_RATE = 0.3       # Higher regularization
EPOCHS = 100
PATIENCE = 20            # More patience for slower learning

# Loss function selection
LOSS_TYPE = 'dice'  # Options: 'dice', 'bce', 'combined'

print("Configuration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Dropout rate: {DROPOUT_RATE}")
print(f"  Loss function: {LOSS_TYPE}")
print(f"  Max epochs: {EPOCHS}")
print()

# ============================================================================
# CUSTOM LOSS FUNCTIONS
# ============================================================================

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss - better for dense object segmentation
    Directly optimizes overlap (related to Jaccard)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice


def combined_loss(y_true, y_pred):
    """
    Combined BCE + Dice loss
    BCE for pixel-wise accuracy, Dice for overlap
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)

    return 0.5 * bce + 0.5 * dice


def get_loss_function(loss_type):
    """Select loss function based on type"""
    if loss_type == 'dice':
        return dice_loss
    elif loss_type == 'bce':
        return 'binary_crossentropy'
    elif loss_type == 'combined':
        return combined_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ============================================================================
# CUSTOM METRICS
# ============================================================================

def dice_coef(y_true, y_pred, smooth=1e-6):
    """Dice coefficient metric"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def jacard_coef(y_true, y_pred, smooth=1e-7):
    """Jaccard coefficient (IoU) metric"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # Binary predictions at 0.5 threshold
    y_pred_binary = K.cast(K.greater(y_pred_f, 0.5), K.floatx())

    intersection = K.sum(y_true_f * y_pred_binary)
    union = K.sum(y_true_f) + K.sum(y_pred_binary) - intersection

    return (intersection + smooth) / (union + smooth)


# ============================================================================
# LOAD DATASET
# ============================================================================

print("Loading dataset...")
image_dataset = []
mask_dataset = []

images = os.listdir(image_directory)
for i, image_name in enumerate(images):
    if image_name.split('.')[-1] in ['tif', 'tiff', 'png', 'jpg']:
        # Load image
        image = cv2.imread(image_directory + image_name, 1)
        if image is None:
            continue
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

        # Load mask
        mask_path = mask_directory + image_name
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0)
            mask = Image.fromarray(mask)
            mask = mask.resize((SIZE, SIZE))
            mask_dataset.append(np.array(mask))

print(f"Loaded {len(image_dataset)} images")

# Normalize
image_dataset = np.array(image_dataset) / 255.
mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

print(f"Image dataset shape: {image_dataset.shape}")
print(f"Mask dataset shape: {mask_dataset.shape}")
print()

# ============================================================================
# STRATIFIED TRAIN/VAL SPLIT
# ============================================================================

print("Creating stratified train/val split...")

# Calculate object density for stratification
object_counts = []
for mask in mask_dataset:
    _, binary_mask = cv2.threshold((mask[:,:,0] * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    num_objects = cv2.connectedComponents(binary_mask)[0] - 1
    object_counts.append(num_objects)

# Create stratification bins
bins = [0, 5, 15, 30, 100]
strata = np.digitize(object_counts, bins)

print(f"Stratification distribution:")
unique, counts = np.unique(strata, return_counts=True)
for stratum, count in zip(unique, counts):
    print(f"  Stratum {stratum}: {count} images")

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    image_dataset, mask_dataset,
    test_size=0.15,  # Slightly larger validation set
    stratify=strata,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_test)}")
print()

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

print("Setting up data augmentation...")

# Aggressive augmentation for circular objects
train_datagen = ImageDataGenerator(
    rotation_range=180,  # Full rotation (beads are circular)
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    fill_mode='reflect'
)

# Seed for reproducibility
seed = 42

print("Augmentation settings:")
print("  - Random rotation: 0-180°")
print("  - Random shifts: ±20%")
print("  - Random flips: horizontal and vertical")
print("  - Random zoom: ±15%")
print("  - Random brightness: 80-120%")
print()

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

print(f"Input shape: {input_shape}")
print()

# ============================================================================
# TRAIN MODELS
# ============================================================================

# Import model architectures
from models import UNet, Attention_UNet, Attention_ResUNet

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"microbead_training_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")
print()

# Training results storage
training_results = []

# Select loss function
loss_function = get_loss_function(LOSS_TYPE)

# ============================================================================
# MODEL 1: STANDARD U-NET
# ============================================================================

print("=" * 80)
print("TRAINING MODEL 1/3: STANDARD U-NET")
print("=" * 80)
print(f"Hyperparameters: LR={LEARNING_RATE}, BS={BATCH_SIZE}, Dropout={DROPOUT_RATE}")
print()

unet_model = UNet(input_shape, dropout_rate=DROPOUT_RATE)
unet_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
    loss=loss_function,
    metrics=['accuracy', jacard_coef, dice_coef]
)

callbacks_unet = [
    EarlyStopping(
        monitor='val_jacard_coef',
        mode='max',
        patience=PATIENCE,
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
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

# Train with augmentation
unet_history = unet_model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=seed),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    callbacks=callbacks_unet,
    verbose=1
)

stop_unet = datetime.now()
execution_time_unet = stop_unet - start_unet

print(f"✓ UNet training completed in {execution_time_unet}")

# Save
unet_model.save(f'{output_dir}/final_unet_model.hdf5')
unet_history_df = pd.DataFrame(unet_history.history)
unet_history_df.to_csv(f'{output_dir}/unet_history.csv', index=False)

best_val_jacard_unet = unet_history_df['val_jacard_coef'].max()
training_results.append({
    'model': 'UNet',
    'lr': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'dropout': DROPOUT_RATE,
    'loss_type': LOSS_TYPE,
    'best_val_jacard': best_val_jacard_unet,
    'training_time': str(execution_time_unet)
})

print(f"Best Val Jaccard: {best_val_jacard_unet:.4f}")
print()

# ============================================================================
# MODEL 2: ATTENTION U-NET
# ============================================================================

print("=" * 80)
print("TRAINING MODEL 2/3: ATTENTION U-NET")
print("=" * 80)
print(f"Hyperparameters: LR={LEARNING_RATE}, BS={BATCH_SIZE}, Dropout={DROPOUT_RATE}")
print()

att_unet_model = Attention_UNet(input_shape, dropout_rate=DROPOUT_RATE)
att_unet_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
    loss=loss_function,
    metrics=['accuracy', jacard_coef, dice_coef]
)

callbacks_att_unet = [
    EarlyStopping(
        monitor='val_jacard_coef',
        mode='max',
        patience=PATIENCE,
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
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
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=seed),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    callbacks=callbacks_att_unet,
    verbose=1
)

stop_att_unet = datetime.now()
execution_time_att_unet = stop_att_unet - start_att_unet

print(f"✓ Attention UNet training completed in {execution_time_att_unet}")

# Save
att_unet_model.save(f'{output_dir}/final_attention_unet_model.hdf5')
att_unet_history_df = pd.DataFrame(att_unet_history.history)
att_unet_history_df.to_csv(f'{output_dir}/attention_unet_history.csv', index=False)

best_val_jacard_att_unet = att_unet_history_df['val_jacard_coef'].max()
training_results.append({
    'model': 'Attention_UNet',
    'lr': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'dropout': DROPOUT_RATE,
    'loss_type': LOSS_TYPE,
    'best_val_jacard': best_val_jacard_att_unet,
    'training_time': str(execution_time_att_unet)
})

print(f"Best Val Jaccard: {best_val_jacard_att_unet:.4f}")
print()

# ============================================================================
# MODEL 3: ATTENTION RESIDUAL U-NET
# ============================================================================

print("=" * 80)
print("TRAINING MODEL 3/3: ATTENTION RESIDUAL U-NET")
print("=" * 80)
print(f"Hyperparameters: LR={LEARNING_RATE}, BS={BATCH_SIZE}, Dropout={DROPOUT_RATE}")
print()

att_res_unet_model = Attention_ResUNet(input_shape, dropout_rate=DROPOUT_RATE)
att_res_unet_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
    loss=loss_function,
    metrics=['accuracy', jacard_coef, dice_coef]
)

callbacks_att_res_unet = [
    EarlyStopping(
        monitor='val_jacard_coef',
        mode='max',
        patience=PATIENCE,
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
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
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=seed),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    callbacks=callbacks_att_res_unet,
    verbose=1
)

stop_att_res_unet = datetime.now()
execution_time_att_res_unet = stop_att_res_unet - start_att_res_unet

print(f"✓ Attention ResUNet training completed in {execution_time_att_res_unet}")

# Save
att_res_unet_model.save(f'{output_dir}/final_attention_resunet_model.hdf5')
att_res_unet_history_df = pd.DataFrame(att_res_unet_history.history)
att_res_unet_history_df.to_csv(f'{output_dir}/attention_resunet_history.csv', index=False)

best_val_jacard_att_res_unet = att_res_unet_history_df['val_jacard_coef'].max()
training_results.append({
    'model': 'Attention_ResUNet',
    'lr': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'dropout': DROPOUT_RATE,
    'loss_type': LOSS_TYPE,
    'best_val_jacard': best_val_jacard_att_res_unet,
    'training_time': str(execution_time_att_res_unet)
})

print(f"Best Val Jaccard: {best_val_jacard_att_res_unet:.4f}")
print()

# ============================================================================
# GENERATE SUMMARY
# ============================================================================

print("=" * 80)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 80)
print()

results_df = pd.DataFrame(training_results)
print(results_df.to_string(index=False))
print()

# Save summary
results_df.to_csv(f'{output_dir}/training_summary.csv', index=False)

# Determine best model
best_model_idx = results_df['best_val_jacard'].idxmax()
best_model = results_df.iloc[best_model_idx]

print("🏆 BEST MODEL:")
print(f"  Architecture: {best_model['model']}")
print(f"  Best Val Jaccard: {best_model['best_val_jacard']:.4f}")
print(f"  Training Time: {best_model['training_time']}")
print()

# Compare with previous mitochondria-optimized results
print("📊 COMPARISON WITH PREVIOUS RESULTS:")
print("  Previous (mitochondria hyperparameters):")
print("    Best Val Jaccard: 0.1427 (collapsed to ~0.0)")
print(f"  Current (microbead hyperparameters):")
print(f"    Best Val Jaccard: {best_model['best_val_jacard']:.4f}")

if best_model['best_val_jacard'] > 0.50:
    print(f"  ✓ EXCELLENT: {best_model['best_val_jacard']/0.1427:.1f}× improvement!")
    print("  → Microbead-optimized hyperparameters successful")
elif best_model['best_val_jacard'] > 0.30:
    print(f"  ✓ GOOD: {best_model['best_val_jacard']/0.1427:.1f}× improvement")
    print("  → Hyperparameters helping, may need further tuning")
elif best_model['best_val_jacard'] > 0.20:
    print(f"  ⚠️  MODERATE: {best_model['best_val_jacard']/0.1427:.1f}× improvement")
    print("  → Check partial mask completion and data quality")
else:
    print("  ❌ POOR: No significant improvement")
    print("  → Check dataset quality and mask completeness")

print()
print("=" * 80)
print(f"All results saved in: {output_dir}/")
print("=" * 80)
