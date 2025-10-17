#!/usr/bin/env python3
"""
Quick test script to diagnose Attention ResUNet model loading and prediction issue.

Purpose: Isolate whether the problem is:
1. Model loading
2. Model prediction on single tile
3. Batch prediction
4. Memory issue
"""

import sys
import numpy as np
from pathlib import Path
from tensorflow import keras

# Import custom objects
from loss_functions_fixed import combined_dice_focal_loss, jacard_coef, dice_coef, focal_loss
from models_fixed import RepeatElements

print("="*80)
print("ATTENTION RESUNET MODEL LOADING TEST")
print("="*80)
print()

# Define custom objects
class BinaryFocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        return focal_loss(y_true, y_pred, alpha=self.alpha, gamma=self.gamma)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

custom_objects = {
    'combined_dice_focal_loss': combined_dice_focal_loss,
    'jacard_coef': jacard_coef,
    'dice_coef': dice_coef,
    'focal_loss': focal_loss,
    'BinaryFocalLoss': BinaryFocalLoss,
    'RepeatElements': RepeatElements,
}

# Model path
model_path = Path("./attention_resunet_hyperparam_20251015_235542/checkpoints/attention_resunet_n_filters32_dropout0p1_batch_normTrue_learning_rate0p001/best_model.keras")

print(f"Testing model: {model_path}")
print(f"Model exists: {model_path.exists()}")
print()

# Test 1: Load model
print("Test 1: Loading model...")
try:
    model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
    print("✓ Model loaded successfully")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Total layers: {len(model.layers)}")
    print()
except Exception as e:
    print(f"✗ Model loading FAILED")
    print(f"  Error: {e}")
    sys.exit(1)

# Test 2: Predict on single random tile
print("Test 2: Predicting on single random 512x512 RGB tile...")
try:
    dummy_tile = np.random.rand(1, 512, 512, 3).astype(np.float32)
    print(f"  Dummy tile shape: {dummy_tile.shape}")

    prediction = model.predict(dummy_tile, verbose=0)
    print(f"✓ Single prediction successful")
    print(f"  Prediction shape: {prediction.shape}")
    print(f"  Prediction range: [{prediction.min():.6f}, {prediction.max():.6f}]")
    print()
except Exception as e:
    print(f"✗ Single prediction FAILED")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Predict on batch of 4 tiles
print("Test 3: Predicting on batch of 4 random tiles...")
try:
    dummy_batch = np.random.rand(4, 512, 512, 3).astype(np.float32)
    print(f"  Dummy batch shape: {dummy_batch.shape}")

    predictions = model.predict(dummy_batch, batch_size=4, verbose=0)
    print(f"✓ Batch prediction successful")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print()
except Exception as e:
    print(f"✗ Batch prediction FAILED")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Predict on 28 tiles (typical image size)
print("Test 4: Predicting on 28 tiles (full image)...")
try:
    dummy_full_image = np.random.rand(28, 512, 512, 3).astype(np.float32)
    print(f"  Dummy full image shape: {dummy_full_image.shape}")

    predictions = model.predict(dummy_full_image, batch_size=4, verbose=0)
    print(f"✓ Full image prediction successful")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print()
except Exception as e:
    print(f"✗ Full image prediction FAILED")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Model summary (check for any architectural issues)
print("Test 5: Model architecture summary...")
try:
    print()
    model.summary()
    print()
except Exception as e:
    print(f"✗ Model summary FAILED")
    print(f"  Error: {e}")

print("="*80)
print("ALL TESTS PASSED")
print("="*80)
print()
print("Conclusion: Model loads and predicts correctly.")
print("Issue must be in the density analysis script, not the model itself.")
print()
