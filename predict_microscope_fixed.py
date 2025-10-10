#!/usr/bin/env python3
"""
Fixed Microbead Prediction Script
==================================
Corrects two critical issues:
1. Tile size matches training (512×512 - using original resolution)
2. No black grid in output (proper blending with smooth transitions)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import argparse
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

# Training parameters (MUST MATCH TRAINING!)
TRAINING_SIZE = 512  # Images trained at original 512×512 resolution
IMG_CHANNELS = 1

def load_model(model_path):
    """Load trained model with custom metrics"""
    from tensorflow.keras import backend as K

    def dice_coef(y_true, y_pred, smooth=1e-6):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def jacard_coef(y_true, y_pred, smooth=1e-6):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)

    def dice_loss(y_true, y_pred, smooth=1e-6):
        return 1 - dice_coef(y_true, y_pred, smooth)

    custom_objects = {
        'dice_coef': dice_coef,
        'jacard_coef': jacard_coef,
        'dice_loss': dice_loss
    }

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"✓ Model loaded successfully")

    return model

class TiledPredictor:
    """
    Predict on large images using tiling strategy with smooth blending.

    Key fixes:
    1. Tile size = 256 (matches training, not 512!)
    2. Gaussian-weighted blending (no black grid)
    3. Proper overlap handling
    """

    def __init__(self, model, tile_size=TRAINING_SIZE, overlap=64):
        """
        Args:
            model: Trained Keras model
            tile_size: Size of tiles (MUST match training size = 512)
            overlap: Pixels of overlap between tiles (creates smooth blending)
        """
        if tile_size != TRAINING_SIZE:
            print(f"WARNING: tile_size ({tile_size}) != training size ({TRAINING_SIZE})")
            print(f"Forcing tile_size = {TRAINING_SIZE} to match training")
            tile_size = TRAINING_SIZE

        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

        # Create Gaussian weight matrix for smooth blending
        self.weight_matrix = self._create_gaussian_weight_matrix()

    def _create_gaussian_weight_matrix(self):
        """
        Create Gaussian weight matrix for smooth tile blending.
        Center pixels have weight 1.0, edges fade to 0.0
        """
        # Create 2D Gaussian
        x = np.linspace(-1, 1, self.tile_size)
        y = np.linspace(-1, 1, self.tile_size)
        X, Y = np.meshgrid(x, y)

        # Gaussian with sigma=0.5 (adjust for more/less smoothness)
        sigma = 0.5
        weight = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

        # Normalize to [0, 1]
        weight = weight / weight.max()

        return weight

    def predict_large_image(self, image, threshold=0.5):
        """
        Predict on image of any size using tiling with smooth blending.

        Args:
            image: Input image (H, W) or (H, W, 1)
            threshold: Threshold for binary mask (default 0.5)

        Returns:
            Binary prediction mask (H, W)
        """
        # Handle dimensions
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image[:, :, 0]

        h, w = image.shape

        # If image is smaller than tile size, resize and predict directly
        if h <= self.tile_size and w <= self.tile_size:
            return self._predict_single_tile(image, threshold)

        # Initialize output arrays
        prediction = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)

        # Calculate number of tiles
        n_tiles_h = int(np.ceil((h - self.overlap) / self.stride))
        n_tiles_w = int(np.ceil((w - self.overlap) / self.stride))

        print(f"  Image size: {h}×{w}")
        print(f"  Tile size: {self.tile_size}×{self.tile_size}")
        print(f"  Overlap: {self.overlap} pixels")
        print(f"  Number of tiles: {n_tiles_h}×{n_tiles_w} = {n_tiles_h * n_tiles_w} tiles")

        # Process each tile
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile coordinates
                y_start = i * self.stride
                x_start = j * self.stride

                y_end = min(y_start + self.tile_size, h)
                x_end = min(x_start + self.tile_size, w)

                # Extract tile
                tile = image[y_start:y_end, x_start:x_end]

                # If tile is smaller than tile_size (edge case), pad it
                if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                    padded_tile = np.zeros((self.tile_size, self.tile_size), dtype=tile.dtype)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile

                # Predict on tile
                tile_pred = self._predict_single_tile(tile, threshold=None)  # Get probabilities

                # Get weight matrix for this tile
                weight = self.weight_matrix.copy()

                # Handle edge tiles (trim weight if needed)
                tile_h = y_end - y_start
                tile_w = x_end - x_start
                if tile_h < self.tile_size or tile_w < self.tile_size:
                    weight = weight[:tile_h, :tile_w]
                    tile_pred = tile_pred[:tile_h, :tile_w]

                # Accumulate weighted prediction
                prediction[y_start:y_end, x_start:x_end] += tile_pred * weight
                weight_sum[y_start:y_end, x_start:x_end] += weight

        # Normalize by weight sum (average overlapping predictions)
        prediction = np.divide(
            prediction,
            weight_sum,
            out=np.zeros_like(prediction),
            where=weight_sum > 0
        )

        # Apply threshold
        prediction_binary = (prediction > threshold).astype(np.uint8)

        return prediction_binary

    def _predict_single_tile(self, tile, threshold=0.5):
        """Predict on a single tile"""
        # Normalize if needed
        if tile.max() > 1.0:
            tile = tile / 255.0

        # Prepare input
        tile_input = tile.astype(np.float32)
        tile_input = np.expand_dims(tile_input, axis=0)  # Add batch dim
        tile_input = np.expand_dims(tile_input, axis=-1)  # Add channel dim

        # Predict
        pred = self.model.predict(tile_input, verbose=0)[0, :, :, 0]

        if threshold is not None:
            pred = (pred > threshold).astype(np.uint8)

        return pred

def predict_on_image(model, image_path, output_dir, tile_size=TRAINING_SIZE, overlap=32, threshold=0.5):
    """Predict on a single image"""

    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    print(f"\nProcessing: {image_path.name}")

    # Create predictor
    predictor = TiledPredictor(model, tile_size=tile_size, overlap=overlap)

    # Predict
    mask = predictor.predict_large_image(image, threshold=threshold)

    # Save prediction
    output_path = output_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(output_path), mask * 255)
    print(f"  ✓ Saved mask: {output_path.name}")

    # Create overlay for visualization
    overlay = create_overlay(image, mask)
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"  ✓ Saved overlay: {overlay_path.name}")

def create_overlay(image, mask, alpha=0.4):
    """Create visualization overlay (image + colored mask)"""

    # Convert grayscale to RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    # Create colored mask (green for positive predictions)
    mask_colored = np.zeros_like(image_rgb)
    mask_colored[:, :, 1] = mask * 255  # Green channel

    # Blend
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, mask_colored, alpha, 0)

    return overlay

def main():
    parser = argparse.ArgumentParser(
        description='Fixed microbead prediction with correct tiling (256×256) and no black grid'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.hdf5)')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: predictions_TIMESTAMP)')
    parser.add_argument('--tile-size', type=int, default=TRAINING_SIZE,
                       help=f'Tile size (default: {TRAINING_SIZE}, MUST match training!)')
    parser.add_argument('--overlap', type=int, default=64,
                       help='Overlap between tiles in pixels (default: 64 for 512×512)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold (default: 0.5)')

    args = parser.parse_args()

    # Validate tile size
    if args.tile_size != TRAINING_SIZE:
        print(f"\n⚠️ WARNING: Specified tile size ({args.tile_size}) != training size ({TRAINING_SIZE})")
        print(f"   Forcing tile size to {TRAINING_SIZE} to match training")
        args.tile_size = TRAINING_SIZE

    print("="*80)
    print("FIXED MICROBEAD PREDICTION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Tile size: {args.tile_size}×{args.tile_size} (original 512×512 resolution ✓)")
    print(f"  Overlap: {args.overlap} pixels (smooth blending ✓)")
    print(f"  Threshold: {args.threshold}")
    print()

    # Load model
    model = load_model(args.model)

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"predictions_{timestamp}")
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")
    print()

    # Find input images
    input_dir = Path(args.input_dir)
    image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(input_dir.glob(f'*{ext}'))
        image_paths.extend(input_dir.glob(f'*{ext.upper()}'))

    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"ERROR: No images found in {input_dir}")
        print(f"Supported formats: {image_extensions}")
        return

    print(f"Found {len(image_paths)} images to process")
    print("="*80)

    # Process each image
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}]")
        predict_on_image(
            model, image_path, output_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
            threshold=args.threshold
        )

    print("\n" + "="*80)
    print("PREDICTION COMPLETE")
    print("="*80)
    print(f"\n✓ Processed {len(image_paths)} images")
    print(f"✓ Output saved to: {output_dir}/")
    print()
    print("Generated files for each image:")
    print("  - <name>_mask.png - Binary prediction mask")
    print("  - <name>_overlay.png - Visualization (image + green mask)")
    print()
    print("Key improvements:")
    print("  ✓ Tile size = 512×512 (matches training at original resolution)")
    print("  ✓ Gaussian blending (no black grid lines)")
    print("  ✓ Smooth transitions between tiles")
    print()

if __name__ == '__main__':
    main()
