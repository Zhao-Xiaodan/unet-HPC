#!/usr/bin/env python3
"""
Microscope Image Prediction Script

Features:
1. Loads best performing model from microscope training
2. Handles large images (>512x512) via tiling strategy
3. Processes entire directories
4. Generates visual comparisons and binary masks

Usage:
    python predict_microscope.py --input_dir ./test_image --output_dir ./predictions

Author: Based on microscope_training_20251008_074915 results
"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 80)
print("MICROSCOPE IMAGE SEGMENTATION - PREDICTION TOOL")
print("=" * 80)
print()


class LargeImagePredictor:
    """Handles prediction on arbitrarily large images via tiling"""

    def __init__(self, model, tile_size=256, overlap=32):
        """
        Args:
            model: Trained Keras model
            tile_size: Size of tiles to process (default 256 for trained models)
            overlap: Overlap between tiles in pixels (reduces edge artifacts)
        """
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

    def predict_large_image(self, image, threshold=0.5):
        """
        Predict segmentation mask for large image using sliding window

        Args:
            image: Input image (H, W, 3) normalized to [0, 1]
            threshold: Binary threshold for mask (default 0.5)

        Returns:
            Binary mask (H, W) with values {0, 255}
        """
        h, w = image.shape[:2]

        # Handle small images directly
        if h <= self.tile_size and w <= self.tile_size:
            return self._predict_single_tile(image, threshold)

        # Initialize output mask and weight map for averaging overlaps
        mask_sum = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        # Create Gaussian weight matrix for smooth blending
        tile_weight = self._create_tile_weights(self.tile_size)

        # Calculate number of tiles needed
        n_tiles_h = int(np.ceil((h - self.overlap) / self.stride))
        n_tiles_w = int(np.ceil((w - self.overlap) / self.stride))

        print(f"  Image size: {h}x{w}")
        print(f"  Processing {n_tiles_h} x {n_tiles_w} = {n_tiles_h * n_tiles_w} tiles...")

        tile_count = 0
        total_tiles = n_tiles_h * n_tiles_w

        # Sliding window iteration
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile boundaries
                y_start = i * self.stride
                x_start = j * self.stride
                y_end = min(y_start + self.tile_size, h)
                x_end = min(x_start + self.tile_size, w)

                # Extract tile
                tile = image[y_start:y_end, x_start:x_end]

                # Pad if necessary (for edge tiles)
                tile_h, tile_w = tile.shape[:2]
                if tile_h < self.tile_size or tile_w < self.tile_size:
                    padded_tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=tile.dtype)
                    padded_tile[:tile_h, :tile_w] = tile
                    tile = padded_tile
                    current_weight = tile_weight[:tile_h, :tile_w]
                else:
                    current_weight = tile_weight

                # Predict on tile
                tile_input = np.expand_dims(tile, 0)
                tile_pred = self.model.predict(tile_input, verbose=0)[0, :, :, 0]

                # Apply threshold and extract valid region
                tile_mask = (tile_pred[:tile_h, :tile_w] > threshold).astype(np.float32)

                # Accumulate with weights
                mask_sum[y_start:y_end, x_start:x_end] += tile_mask * current_weight
                weight_map[y_start:y_end, x_start:x_end] += current_weight

                tile_count += 1
                if tile_count % 10 == 0 or tile_count == total_tiles:
                    print(f"    Progress: {tile_count}/{total_tiles} tiles ({tile_count/total_tiles*100:.1f}%)")

        # Normalize by weights (average overlapping predictions)
        mask_normalized = mask_sum / (weight_map + 1e-7)

        # Convert to binary mask
        binary_mask = (mask_normalized > 0.5).astype(np.uint8) * 255

        return binary_mask

    def _predict_single_tile(self, image, threshold=0.5):
        """Predict on image that fits in one tile"""
        h, w = image.shape[:2]

        # Pad to tile_size if needed
        if h < self.tile_size or w < self.tile_size:
            padded = np.zeros((self.tile_size, self.tile_size, 3), dtype=image.dtype)
            padded[:h, :w] = image
            image = padded

        # Predict
        image_input = np.expand_dims(image, 0)
        pred = self.model.predict(image_input, verbose=0)[0, :, :, 0]

        # Threshold and crop to original size
        binary_mask = (pred[:h, :w] > threshold).astype(np.uint8) * 255

        return binary_mask

    def _create_tile_weights(self, size):
        """
        Create Gaussian-like weight matrix for smooth tile blending
        Higher weights in center, lower at edges
        """
        center = size // 2
        y, x = np.ogrid[:size, :size]

        # Distance from center
        dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
        max_dist = np.sqrt(2) * center

        # Gaussian-like falloff
        weights = np.exp(-3 * (dist_from_center / max_dist)**2)

        return weights


def load_best_model(training_dir):
    """
    Load the best performing model from training directory

    Args:
        training_dir: Path to training output directory

    Returns:
        model: Loaded Keras model
        model_name: Name of the best model
    """
    # Based on analysis: Attention ResUNet achieved best peak (0.1427)
    # But all models collapsed, so we load from best epoch (epoch 1)

    # Priority order: Attention ResUNet > UNet > Attention UNet
    model_candidates = [
        ('best_attention_resunet_model.hdf5', 'Attention ResUNet'),
        ('best_unet_model.hdf5', 'UNet'),
        ('best_attention_unet_model.hdf5', 'Attention UNet'),
    ]

    print("Searching for best model...")
    for model_file, model_name in model_candidates:
        model_path = os.path.join(training_dir, model_file)
        if os.path.exists(model_path):
            print(f"  Found: {model_name} at {model_path}")
            print(f"  Loading model...")

            # Import custom metrics
            from models import jacard_coef
            from focal_loss import BinaryFocalLoss

            # Load model with custom objects
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'jacard_coef': jacard_coef,
                    'BinaryFocalLoss': BinaryFocalLoss(gamma=2)
                },
                compile=False
            )

            print(f"  ✓ Model loaded successfully!")
            print(f"  Model: {model_name}")
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
            print()

            return model, model_name

    raise FileNotFoundError(f"No model files found in {training_dir}")


def load_and_preprocess_image(image_path):
    """
    Load and preprocess image for prediction

    Args:
        image_path: Path to input image

    Returns:
        image_normalized: Image normalized to [0, 1]
        image_original: Original image for visualization
    """
    # Read image
    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0

    return img_normalized, img_rgb


def save_predictions(image_name, original_image, predicted_mask, output_dir,
                    save_overlay=True, save_binary=True, save_comparison=True):
    """
    Save prediction results in multiple formats

    Args:
        image_name: Original image filename
        original_image: Original RGB image
        predicted_mask: Binary mask (0-255)
        output_dir: Output directory
        save_overlay: Save overlay visualization
        save_binary: Save binary mask as PNG
        save_comparison: Save side-by-side comparison
    """
    base_name = Path(image_name).stem

    # Create output subdirectories
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)

    # 1. Save binary mask
    if save_binary:
        mask_path = os.path.join(output_dir, 'masks', f'{base_name}_mask.png')
        cv2.imwrite(mask_path, predicted_mask)
        print(f"  Saved binary mask: {mask_path}")

    # 2. Save overlay (mask colored on original image)
    if save_overlay:
        overlay = original_image.copy()
        # Create colored mask (green for mitochondria)
        mask_colored = np.zeros_like(original_image)
        mask_colored[:, :, 1] = predicted_mask  # Green channel

        # Blend with original (50% transparency)
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

        overlay_path = os.path.join(output_dir, 'overlays', f'{base_name}_overlay.png')
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  Saved overlay: {overlay_path}")

    # 3. Save comparison figure
    if save_comparison:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(predicted_mask, cmap='gray')
        axes[1].set_title('Predicted Mask', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # Overlay
        overlay = original_image.copy()
        mask_colored = np.zeros_like(original_image)
        mask_colored[:, :, 1] = predicted_mask
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Green = Mitochondria)', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        plt.suptitle(f'Segmentation Result: {image_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        comparison_path = os.path.join(output_dir, 'comparisons', f'{base_name}_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved comparison: {comparison_path}")

    # Calculate and print statistics
    total_pixels = predicted_mask.size
    positive_pixels = np.sum(predicted_mask > 0)
    coverage_percent = (positive_pixels / total_pixels) * 100

    print(f"  Statistics:")
    print(f"    Image size: {predicted_mask.shape[1]}x{predicted_mask.shape[0]}")
    print(f"    Predicted mitochondria coverage: {coverage_percent:.2f}%")
    print(f"    Positive pixels: {positive_pixels:,} / {total_pixels:,}")


def process_directory(input_dir, output_dir, model, model_name, tile_size=256, overlap=32):
    """
    Process all images in a directory

    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for predictions
        model: Loaded Keras model
        model_name: Name of the model
        tile_size: Tile size for large images
        overlap: Overlap between tiles
    """
    # Create predictor
    predictor = LargeImagePredictor(model, tile_size=tile_size, overlap=overlap)

    # Find all images
    supported_formats = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    image_files = []

    for ext in supported_formats:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))

    if not image_files:
        print(f"ERROR: No images found in {input_dir}")
        print(f"Supported formats: {', '.join(supported_formats)}")
        return

    print(f"Found {len(image_files)} images to process")
    print(f"Using model: {model_name}")
    print(f"Tile size: {tile_size}x{tile_size}, Overlap: {overlap}px")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each image
    results = []
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")

        try:
            # Load image
            img_normalized, img_original = load_and_preprocess_image(image_path)

            # Predict
            start_time = datetime.now()
            predicted_mask = predictor.predict_large_image(img_normalized, threshold=0.5)
            elapsed = (datetime.now() - start_time).total_seconds()

            print(f"  Prediction time: {elapsed:.2f}s")

            # Save results
            save_predictions(
                image_path.name,
                img_original,
                predicted_mask,
                output_dir,
                save_overlay=True,
                save_binary=True,
                save_comparison=True
            )

            # Record result
            results.append({
                'filename': image_path.name,
                'size': f"{img_original.shape[1]}x{img_original.shape[0]}",
                'coverage': f"{(np.sum(predicted_mask > 0) / predicted_mask.size) * 100:.2f}%",
                'time': f"{elapsed:.2f}s",
                'status': 'Success'
            })

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({
                'filename': image_path.name,
                'size': 'N/A',
                'coverage': 'N/A',
                'time': 'N/A',
                'status': f'Failed: {str(e)}'
            })

        print()

    # Save processing summary
    save_summary(results, output_dir, model_name)

    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Processed: {len(image_files)} images")
    print(f"Output directory: {output_dir}")
    print(f"  - Binary masks: {output_dir}/masks/")
    print(f"  - Overlays: {output_dir}/overlays/")
    print(f"  - Comparisons: {output_dir}/comparisons/")
    print(f"  - Summary: {output_dir}/prediction_summary.txt")


def save_summary(results, output_dir, model_name):
    """Save processing summary to text file"""
    summary_path = os.path.join(output_dir, 'prediction_summary.txt')

    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MICROSCOPE IMAGE SEGMENTATION - PREDICTION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model: {model_name}\n")
        f.write(f"Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total images: {len(results)}\n\n")

        f.write("Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Filename':<40} {'Size':<12} {'Coverage':<12} {'Time':<10} {'Status':<20}\n")
        f.write("-" * 80 + "\n")

        for r in results:
            f.write(f"{r['filename']:<40} {r['size']:<12} {r['coverage']:<12} {r['time']:<10} {r['status']:<20}\n")

        successful = sum(1 for r in results if r['status'] == 'Success')
        f.write("-" * 80 + "\n")
        f.write(f"\nSuccessful: {successful}/{len(results)}\n")

    print(f"Saved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict mitochondria segmentation masks from microscope images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process test images with default settings
  python predict_microscope.py --input_dir ./test_image

  # Custom output directory
  python predict_microscope.py --input_dir ./test_image --output_dir ./my_predictions

  # Adjust tile overlap for large images
  python predict_microscope.py --input_dir ./test_image --overlap 64

  # Use specific training directory
  python predict_microscope.py --input_dir ./test_image --model_dir ./my_training_dir
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                       help='Output directory for predictions (default: ./predictions)')
    parser.add_argument('--model_dir', type=str,
                       default='microscope_training_20251008_074915',
                       help='Training directory containing model files')
    parser.add_argument('--tile_size', type=int, default=256,
                       help='Tile size for processing large images (default: 256)')
    parser.add_argument('--overlap', type=int, default=32,
                       help='Overlap between tiles in pixels (default: 32)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Binary threshold for mask (default: 0.5)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}")
        return

    if not os.path.exists(args.model_dir):
        print(f"ERROR: Model directory not found: {args.model_dir}")
        return

    # Load model
    try:
        model, model_name = load_best_model(args.model_dir)
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        return

    # Process directory
    process_directory(
        args.input_dir,
        args.output_dir,
        model,
        model_name,
        tile_size=args.tile_size,
        overlap=args.overlap
    )


if __name__ == '__main__':
    main()
