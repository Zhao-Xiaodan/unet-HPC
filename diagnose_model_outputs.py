#!/usr/bin/env python3
"""
Diagnose Model Output Issues
=============================
Investigates why models are producing incorrect predictions by examining
raw model outputs before thresholding.
"""

import os
import sys
import numpy as np
import cv2
import glob
from pathlib import Path
from tensorflow import keras
from model_architectures import get_model
from loss_functions import jacard_coef, dice_coef, get_loss_function

# Configuration
TEST_IMAGE = './test_images/10x_2025-05-15_02-05-00.tif'
MODELS_DIR = './hyperparam_comprehensive_20251012_005054'


def find_model_file(models_dir, architecture):
    """Find model file for given architecture"""
    patterns = [
        f"model_{architecture}_bs8_dr0.3_combined_tversky.hdf5",
        f"model_{architecture}_bs8_dr0.3_combined.hdf5",
        f"model_{architecture}_bs8_*.hdf5",
        f"model_{architecture}_*.hdf5"
    ]

    for pattern in patterns:
        matches = glob.glob(os.path.join(models_dir, pattern))
        if matches:
            return matches[0]

    return None


def load_test_tile(image_path):
    """Load test image and extract first 512x512 tile"""
    print(f"\nLoading test image: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load: {image_path}")

    print(f"  Image shape: {img.shape}")
    print(f"  Image range: [{img.min()}, {img.max()}]")

    # Extract first tile
    tile = img[0:512, 0:512]

    # Normalize to [0, 1]
    tile_norm = tile.astype(np.float32) / 255.0

    # Add batch and channel dimensions
    tile_input = tile_norm[np.newaxis, :, :, np.newaxis]

    print(f"  Tile shape: {tile_input.shape}")
    print(f"  Tile range after normalization: [{tile_input.min():.6f}, {tile_input.max():.6f}]")

    return tile, tile_input


def analyze_model_output(architecture, model_path, tile_input):
    """Analyze raw model output"""
    print(f"\n{'='*80}")
    print(f"{architecture.upper()} MODEL ANALYSIS")
    print(f"{'='*80}")
    print(f"Model file: {Path(model_path).name}")

    # Custom objects for loading
    custom_objects = {
        'jacard_coef': jacard_coef,
        'dice_coef': dice_coef,
        'combined_tversky_focal_loss': get_loss_function('combined_tversky'),
        'combined_dice_focal_loss': get_loss_function('combined'),
        'focal_loss': get_loss_function('focal'),
        'focal_tversky_loss': get_loss_function('focal_tversky'),
        'tversky_loss': get_loss_function('tversky')
    }

    try:
        # Load model
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✓ Model loaded successfully")

        # Get model summary
        print(f"\nModel architecture:")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Total parameters: {model.count_params():,}")

        # Check output layer
        output_layer = model.layers[-1]
        print(f"  Output layer: {output_layer.name} ({output_layer.__class__.__name__})")
        if hasattr(output_layer, 'activation'):
            print(f"  Output activation: {output_layer.activation.__name__}")

        # Get prediction
        print(f"\nRunning prediction...")
        pred = model.predict(tile_input, verbose=0)

        # Analyze prediction
        print(f"\nRaw Prediction Statistics:")
        print(f"  Shape: {pred.shape}")
        print(f"  Data type: {pred.dtype}")
        print(f"  Min value: {pred.min():.6f}")
        print(f"  Max value: {pred.max():.6f}")
        print(f"  Mean value: {pred.mean():.6f}")
        print(f"  Median value: {np.median(pred):.6f}")
        print(f"  Std dev: {pred.std():.6f}")

        # Analyze distribution
        pred_flat = pred.flatten()
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            val = np.percentile(pred_flat, p)
            print(f"  {p:2d}th: {val:.6f}")

        # Check threshold behaviors
        print(f"\nThreshold Analysis:")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for thresh in thresholds:
            pct = (pred > thresh).sum() / pred.size * 100
            print(f"  > {thresh:.1f}: {pct:6.2f}% pixels")

        # Apply Otsu threshold
        pred_uint8 = (pred[0, :, :, 0] * 255).astype(np.uint8)
        otsu_thresh, binary_otsu = cv2.threshold(
            pred_uint8, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        otsu_thresh_norm = otsu_thresh / 255.0
        density_otsu = (binary_otsu > 0).sum() / binary_otsu.size

        print(f"\nOtsu Thresholding:")
        print(f"  Optimal threshold: {otsu_thresh_norm:.4f}")
        print(f"  Resulting density: {density_otsu:.4f} ({density_otsu*100:.2f}%)")

        # Standard 0.5 threshold
        binary_fixed = (pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        density_fixed = (binary_fixed > 0).sum() / binary_fixed.size

        print(f"\nFixed 0.5 Threshold:")
        print(f"  Resulting density: {density_fixed:.4f} ({density_fixed*100:.2f}%)")

        # Diagnosis
        print(f"\n{'='*80}")
        print("DIAGNOSIS")
        print(f"{'='*80}")

        if pred.max() < 0.1:
            print("⚠ ISSUE: Output values very low (max < 0.1)")
            print("  → Model is predicting mostly background everywhere")
            print("  → Possible causes:")
            print("     - Model not trained properly")
            print("     - Wrong model loaded")
            print("     - Input normalization mismatch")
            print(f"  → Recommended threshold: {otsu_thresh_norm:.3f} (Otsu)")

        elif pred.min() > 0.9:
            print("⚠ ISSUE: Output values very high (min > 0.9)")
            print("  → Model is predicting mostly foreground everywhere")
            print("  → Possible causes:")
            print("     - Model weights corrupted")
            print("     - Architecture issue")
            print("     - Wrong final activation")

        elif pred.mean() > 0.9:
            print("⚠ ISSUE: Mean output > 0.9")
            print("  → Model heavily biased toward foreground prediction")
            print("  → This will result in almost all-white masks")

        elif pred.mean() < 0.1:
            print("⚠ ISSUE: Mean output < 0.1")
            print("  → Model heavily biased toward background prediction")
            print("  → This will result in almost all-black masks")
            print(f"  → Recommended: Lower threshold to {otsu_thresh_norm:.3f}")

        elif density_fixed < 0.01:
            print("⚠ ISSUE: With threshold 0.5, predicted density < 1%")
            print("  → Too few pixels above threshold")
            print(f"  → Recommended: Use Otsu threshold ({otsu_thresh_norm:.3f})")

        elif density_fixed > 0.99:
            print("⚠ ISSUE: With threshold 0.5, predicted density > 99%")
            print("  → Too many pixels above threshold")
            print(f"  → Recommended: Increase threshold to {otsu_thresh_norm:.3f}")

        else:
            print("✓ Output appears reasonable")
            print(f"  Mean: {pred.mean():.4f}, Density @ 0.5: {density_fixed*100:.2f}%")

        return {
            'model': model,
            'prediction': pred,
            'stats': {
                'min': float(pred.min()),
                'max': float(pred.max()),
                'mean': float(pred.mean()),
                'std': float(pred.std()),
                'density_fixed': float(density_fixed),
                'density_otsu': float(density_otsu),
                'otsu_threshold': float(otsu_thresh_norm)
            }
        }

    except Exception as e:
        print(f"✗ Error analyzing {architecture}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*80)
    print("MODEL OUTPUT DIAGNOSTIC TOOL")
    print("="*80)

    # Check test image exists
    if not os.path.exists(TEST_IMAGE):
        print(f"✗ Test image not found: {TEST_IMAGE}")
        print("  Please update TEST_IMAGE path in script")
        sys.exit(1)

    # Load test tile
    try:
        tile_original, tile_input = load_test_tile(TEST_IMAGE)
    except Exception as e:
        print(f"✗ Error loading test image: {e}")
        sys.exit(1)

    # Calculate reference density with CLAHE+OTSU
    print(f"\n{'='*80}")
    print("REFERENCE: CLAHE+OTSU METHOD")
    print(f"{'='*80}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(tile_original)
    _, binary_ref = cv2.threshold(clahe_img, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    density_ref = (binary_ref > 0).sum() / binary_ref.size

    print(f"Reference density (CLAHE+OTSU): {density_ref:.4f} ({density_ref*100:.2f}%)")

    # Analyze each architecture
    results = {}
    for arch in ['unet', 'resunet', 'attention_resunet']:
        model_path = find_model_file(MODELS_DIR, arch)

        if model_path is None:
            print(f"\n✗ No model file found for {arch}")
            continue

        result = analyze_model_output(arch, model_path, tile_input)
        if result:
            results[arch] = result

    # Summary comparison
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON")
        print(f"{'='*80}")

        print(f"\n{'Architecture':<20s} {'Mean Output':>12s} {'Density@0.5':>12s} {'Otsu Thresh':>12s} {'Density@Otsu':>12s}")
        print("-" * 80)
        print(f"{'CLAHE+OTSU (ref)':<20s} {'N/A':>12s} {'N/A':>12s} {'N/A':>12s} {density_ref:>12.4f}")

        for arch, result in results.items():
            stats = result['stats']
            print(f"{arch:<20s} {stats['mean']:>12.4f} {stats['density_fixed']:>12.4f} "
                  f"{stats['otsu_threshold']:>12.4f} {stats['density_otsu']:>12.4f}")

        # Recommendations
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")

        all_good = True
        for arch, result in results.items():
            stats = result['stats']

            if stats['mean'] > 0.9:
                print(f"\n{arch.upper()}:")
                print(f"  ✗ Model predicting ~100% foreground - DO NOT USE")
                print(f"  → Retrain this model or use different checkpoint")
                all_good = False

            elif stats['density_fixed'] < 0.01:
                print(f"\n{arch.upper()}:")
                print(f"  ⚠ Under-segmenting with 0.5 threshold")
                print(f"  → Use Otsu threshold: {stats['otsu_threshold']:.3f}")
                print(f"  → Expected density: {stats['density_otsu']:.2%} vs reference {density_ref:.2%}")
                all_good = False

            elif abs(stats['density_otsu'] - density_ref) / density_ref > 0.5:
                print(f"\n{arch.upper()}:")
                print(f"  ⚠ Large deviation from reference even with Otsu")
                print(f"  → Predicted: {stats['density_otsu']:.2%}, Reference: {density_ref:.2%}")
                print(f"  → Model may need retraining")
                all_good = False

        if all_good:
            print("\n✓ All models producing reasonable outputs")
            print("  Use Otsu thresholding for best results")

    print(f"\n{'='*80}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
