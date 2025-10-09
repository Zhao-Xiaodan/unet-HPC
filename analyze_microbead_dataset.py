#!/usr/bin/env python3
"""
Microbead Dataset Analysis

Analyzes dataset characteristics to determine optimal hyperparameters:
- Object density (beads per image)
- Class balance (positive vs negative pixels)
- Object size distribution
- Partial mask detection
- Comparison with mitochondria dataset characteristics

Usage:
    python analyze_microbead_dataset.py
"""

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json

print("=" * 80)
print("MICROBEAD DATASET ANALYSIS")
print("=" * 80)
print()

# Dataset paths
image_directory = 'dataset_microscope/images/'
mask_directory = 'dataset_microscope/masks/'

SIZE = 256

# Load dataset
print("Loading dataset...")
image_dataset = []
mask_dataset = []
filenames = []

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
            if mask is not None:
                mask = Image.fromarray(mask)
                mask = mask.resize((SIZE, SIZE))  # Use mask, not image
                mask_dataset.append(np.array(mask))
                filenames.append(image_name)

print(f"Loaded {len(image_dataset)} images with masks")
print()

# Convert to arrays
images_array = np.array(image_dataset)
masks_array = np.array(mask_dataset)

# Normalize
masks_normalized = masks_array / 255.0

# ============================================================================
# ANALYSIS 1: Class Balance
# ============================================================================

print("=" * 80)
print("ANALYSIS 1: CLASS BALANCE")
print("=" * 80)

positive_ratios = []
for mask in masks_normalized:
    positive_pixels = np.sum(mask > 0.5)
    total_pixels = mask.size
    ratio = positive_pixels / total_pixels
    positive_ratios.append(ratio)

mean_positive_ratio = np.mean(positive_ratios)
std_positive_ratio = np.std(positive_ratios)

print(f"Mean positive class ratio: {mean_positive_ratio:.3f} ({mean_positive_ratio*100:.1f}%)")
print(f"Std positive class ratio: {std_positive_ratio:.3f}")
print(f"Min: {np.min(positive_ratios):.3f}, Max: {np.max(positive_ratios):.3f}")
print()

# Compare with mitochondria dataset
print("COMPARISON WITH MITOCHONDRIA DATASET:")
print("  Mitochondria: ~10-15% positive pixels")
print(f"  Microbeads: {mean_positive_ratio*100:.1f}% positive pixels")

if mean_positive_ratio > 0.3:
    print("  ⚠️  CRITICAL: Class balance is VERY DIFFERENT from mitochondria!")
    print("  → Focal Loss (γ=2) is INAPPROPRIATE for this dataset")
    print("  → Recommendation: Use Dice Loss or Binary Cross-Entropy")
elif mean_positive_ratio > 0.2:
    print("  ⚠️  WARNING: Significantly more balanced than mitochondria")
    print("  → Recommendation: Reduce focal gamma to 1.0 or use Dice Loss")
else:
    print("  ✓ Similar class balance to mitochondria")

print()

# ============================================================================
# ANALYSIS 2: Object Density
# ============================================================================

print("=" * 80)
print("ANALYSIS 2: OBJECT DENSITY (Beads per Image)")
print("=" * 80)

object_counts = []
object_sizes = []

for i, mask in enumerate(masks_array):
    # Ensure mask is uint8 and single channel
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]  # Take first channel if multi-channel

    mask_uint8 = mask.astype(np.uint8)

    # Threshold mask
    _, binary_mask = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

    # Count connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # Exclude background (label 0)
    num_objects = num_labels - 1
    object_counts.append(num_objects)

    # Get object sizes (excluding background)
    if num_objects > 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        object_sizes.extend(areas)

mean_objects = np.mean(object_counts)
std_objects = np.std(object_counts)

print(f"Mean objects per image: {mean_objects:.1f}")
print(f"Std objects per image: {std_objects:.1f}")
print(f"Min: {np.min(object_counts)}, Max: {np.max(object_counts)}")
print()

# Compare with mitochondria
print("COMPARISON WITH MITOCHONDRIA DATASET:")
print("  Mitochondria: ~2-3 objects per image")
print(f"  Microbeads: {mean_objects:.1f} objects per image")

if mean_objects > 10:
    print(f"  ⚠️  CRITICAL: {mean_objects/3:.1f}× MORE DENSE than mitochondria!")
    print("  → Requires DIFFERENT hyperparameters:")
    print("     • Lower learning rate (1e-4 instead of 1e-3)")
    print("     • Larger batch size (32 instead of 8)")
    print("     • More regularization (dropout 0.3)")
elif mean_objects > 5:
    print(f"  ⚠️  WARNING: {mean_objects/3:.1f}× more dense than mitochondria")
    print("  → Recommendation: Adjust learning rate and batch size")
else:
    print("  ✓ Similar density to mitochondria")

print()

# Object size distribution
if len(object_sizes) > 0:
    print(f"Object size statistics:")
    print(f"  Mean area: {np.mean(object_sizes):.1f} pixels")
    print(f"  Median area: {np.median(object_sizes):.1f} pixels")
    print(f"  Std area: {np.std(object_sizes):.1f} pixels")
    print(f"  Min area: {np.min(object_sizes)}, Max area: {np.max(object_sizes)}")
print()

# ============================================================================
# ANALYSIS 3: Partial Mask Detection
# ============================================================================

print("=" * 80)
print("ANALYSIS 3: PARTIAL MASK DETECTION")
print("=" * 80)

print("Analyzing potential partial labeling issues...")
print()

partial_mask_suspects = []

for i, (image, mask, filename) in enumerate(zip(images_array, masks_array, filenames)):
    # Simple detection: Look for beads in image that aren't in mask

    # Convert image to uint8 for processing
    image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    # Convert to grayscale
    if len(image_uint8.shape) == 3:
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_uint8

    # Detect circular objects in image (simple approach)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=50,
        param2=30,
        minRadius=3,
        maxRadius=30
    )

    num_detected_circles = 0 if circles is None else len(circles[0])

    # Count labeled objects in mask
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    mask_uint8 = mask.astype(np.uint8)
    _, binary_mask = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
    num_labels = cv2.connectedComponents(binary_mask)[0] - 1

    # Check if significantly fewer labels than detected circles
    if num_detected_circles > 0:
        coverage_ratio = num_labels / num_detected_circles

        if coverage_ratio < 0.7:
            partial_mask_suspects.append({
                'filename': filename,
                'detected_circles': num_detected_circles,
                'labeled_objects': num_labels,
                'coverage_ratio': coverage_ratio
            })

print(f"Images analyzed: {len(filenames)}")
print(f"Potential partial masks detected: {len(partial_mask_suspects)}")
print()

if len(partial_mask_suspects) > 0:
    print("⚠️  WARNING: Partial mask labeling detected!")
    print(f"  {len(partial_mask_suspects)}/{len(filenames)} images ({len(partial_mask_suspects)/len(filenames)*100:.1f}%) may have incomplete labels")
    print()
    print("Top 5 suspected partial masks:")
    sorted_suspects = sorted(partial_mask_suspects, key=lambda x: x['coverage_ratio'])[:5]
    for suspect in sorted_suspects:
        print(f"  {suspect['filename']}: {suspect['labeled_objects']}/{suspect['detected_circles']} labeled ({suspect['coverage_ratio']*100:.1f}%)")
    print()
    print("  ⚠️  This is CRITICAL and likely the main cause of validation collapse!")
    print("  → URGENT: Complete the mask annotations before retraining")
    print("  → Suggested: Use current predictions to identify missing labels")
else:
    print("✓ No obvious partial mask issues detected")
    print("  (Note: This is a heuristic check, manual verification recommended)")

print()

# ============================================================================
# ANALYSIS 4: Train/Val Split Recommendation
# ============================================================================

print("=" * 80)
print("ANALYSIS 4: TRAIN/VAL SPLIT STRATIFICATION")
print("=" * 80)

# Group images by object density
bins = [0, 5, 15, 30, 100]
bin_labels = ['Very sparse (0-5)', 'Sparse (6-15)', 'Medium (16-30)', 'Dense (31+)']

hist, bin_edges = np.histogram(object_counts, bins=bins)

print("Object density distribution:")
for label, count in zip(bin_labels, hist):
    print(f"  {label}: {count} images ({count/len(object_counts)*100:.1f}%)")
print()

# Check if distribution is balanced
min_bin = np.min(hist[hist > 0])
max_bin = np.max(hist)

if max_bin / min_bin > 3:
    print("⚠️  WARNING: Unbalanced density distribution")
    print(f"  → Recommendation: Use stratified train/val split")
    print(f"  → Code: stratify=np.digitize(object_counts, bins={bins})")
else:
    print("✓ Reasonable distribution balance")
    print("  → Standard random split should work")

print()

# ============================================================================
# ANALYSIS 5: Image Quality Assessment
# ============================================================================

print("=" * 80)
print("ANALYSIS 5: IMAGE QUALITY ASSESSMENT")
print("=" * 80)

brightness_values = []
contrast_values = []

for image in images_array:
    # Convert to uint8 for consistent processing
    image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    if len(image_uint8.shape) == 3:
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_uint8

    # Brightness (mean intensity)
    brightness = np.mean(gray)
    brightness_values.append(brightness)

    # Contrast (std of intensity)
    contrast = np.std(gray)
    contrast_values.append(contrast)

print(f"Brightness statistics:")
print(f"  Mean: {np.mean(brightness_values):.1f}")
print(f"  Std: {np.std(brightness_values):.1f}")
print(f"  Range: [{np.min(brightness_values):.1f}, {np.max(brightness_values):.1f}]")
print()

print(f"Contrast statistics:")
print(f"  Mean: {np.mean(contrast_values):.1f}")
print(f"  Std: {np.std(contrast_values):.1f}")
print(f"  Range: [{np.min(contrast_values):.1f}, {np.max(contrast_values):.1f}]")
print()

if np.std(brightness_values) > 30:
    print("⚠️  WARNING: High brightness variation across images")
    print("  → Recommendation: Add brightness augmentation during training")
elif np.std(brightness_values) > 20:
    print("ℹ  Moderate brightness variation")
    print("  → Recommendation: Consider brightness normalization")
else:
    print("✓ Consistent brightness across dataset")

print()

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Create output directory
os.makedirs('dataset_analysis', exist_ok=True)

# Figure 1: Class balance distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1.1 Positive class ratio histogram
axes[0, 0].hist(positive_ratios, bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(mean_positive_ratio, color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_positive_ratio:.3f}')
axes[0, 0].axvline(0.10, color='g', linestyle=':', linewidth=2,
                   label='Mitochondria (~0.10)')
axes[0, 0].set_xlabel('Positive Class Ratio')
axes[0, 0].set_ylabel('Number of Images')
axes[0, 0].set_title('Positive Class Distribution', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 1.2 Object count histogram
axes[0, 1].hist(object_counts, bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(mean_objects, color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_objects:.1f}')
axes[0, 1].axvline(2.5, color='g', linestyle=':', linewidth=2,
                   label='Mitochondria (~2.5)')
axes[0, 1].set_xlabel('Objects per Image')
axes[0, 1].set_ylabel('Number of Images')
axes[0, 1].set_title('Object Density Distribution', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 1.3 Object size distribution
if len(object_sizes) > 0:
    axes[1, 0].hist(object_sizes, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Object Area (pixels)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Object Size Distribution', fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

# 1.4 Brightness vs Contrast scatter
axes[1, 1].scatter(brightness_values, contrast_values, alpha=0.5)
axes[1, 1].set_xlabel('Mean Brightness')
axes[1, 1].set_ylabel('Contrast (Std Dev)')
axes[1, 1].set_title('Image Quality Distribution', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dataset_analysis/distribution_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: dataset_analysis/distribution_analysis.png")

# Figure 2: Sample images with annotations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

sample_indices = np.linspace(0, len(images_array)-1, 6, dtype=int)

for idx, ax_idx in enumerate(sample_indices):
    row = idx // 3
    col = idx % 3

    # Show image with mask overlay
    img = images_array[ax_idx]
    mask = masks_array[ax_idx]

    # Ensure proper format
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    # Create overlay
    overlay = img.copy()
    mask_colored = np.zeros_like(overlay)

    # Handle mask format
    if len(mask.shape) > 2:
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask

    mask_colored[:, :, 1] = mask_2d  # Green channel
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

    axes[row, col].imshow(overlay)
    axes[row, col].set_title(
        f"{filenames[ax_idx]}\nObjects: {object_counts[ax_idx]}, "
        f"Positive: {positive_ratios[ax_idx]*100:.1f}%",
        fontsize=9
    )
    axes[row, col].axis('off')

plt.suptitle('Sample Images with Mask Overlays', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dataset_analysis/sample_images.png', dpi=300, bbox_inches='tight')
print("Saved: dataset_analysis/sample_images.png")

plt.close('all')

# ============================================================================
# SAVE ANALYSIS SUMMARY
# ============================================================================

summary = {
    'dataset_size': len(images_array),
    'mean_positive_ratio': float(mean_positive_ratio),
    'std_positive_ratio': float(std_positive_ratio),
    'mean_objects_per_image': float(mean_objects),
    'std_objects_per_image': float(std_objects),
    'mean_object_size': float(np.mean(object_sizes)) if len(object_sizes) > 0 else 0,
    'partial_mask_suspects': len(partial_mask_suspects),
    'partial_mask_ratio': len(partial_mask_suspects) / len(images_array) if len(images_array) > 0 else 0,
    'mean_brightness': float(np.mean(brightness_values)),
    'std_brightness': float(np.std(brightness_values))
}

with open('dataset_analysis/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Saved: dataset_analysis/summary.json")
print()

# ============================================================================
# RECOMMENDATIONS SUMMARY
# ============================================================================

print("=" * 80)
print("RECOMMENDATIONS SUMMARY")
print("=" * 80)
print()

print("Based on analysis, recommended changes from mitochondria settings:")
print()

# Learning rate
if mean_objects > 10:
    print("1. LEARNING RATE:")
    print("   Current (mitochondria): 1e-3 (UNet), 1e-4 (Attention models)")
    print("   Recommended (microbeads): 5e-5 to 1e-4")
    print(f"   Reason: {mean_objects/3:.1f}× more dense → stronger gradients")
    print()

# Batch size
if mean_objects > 5:
    print("2. BATCH SIZE:")
    print("   Current (mitochondria): 8-16")
    print("   Recommended (microbeads): 32-64")
    print("   Reason: Dense objects → need more samples for gradient stability")
    print()

# Loss function
if mean_positive_ratio > 0.25:
    print("3. LOSS FUNCTION:")
    print("   Current (mitochondria): Binary Focal Loss (γ=2)")
    print("   Recommended (microbeads): Dice Loss or Binary Cross-Entropy")
    print(f"   Reason: {mean_positive_ratio*100:.1f}% positive vs 10-15% → more balanced")
    print()

# Regularization
print("4. REGULARIZATION:")
print("   Current (mitochondria): Minimal dropout")
print("   Recommended (microbeads): Dropout 0.3, L2 weight decay 1e-4")
print("   Reason: Dense uniform objects → higher overfitting risk")
print()

# Partial masks
if len(partial_mask_suspects) > len(images_array) * 0.1:
    print("⚠️  5. CRITICAL: PARTIAL MASK ISSUE")
    print(f"   {len(partial_mask_suspects)} images ({len(partial_mask_suspects)/len(images_array)*100:.1f}%) have incomplete labels")
    print("   *** MUST FIX BEFORE RETRAINING ***")
    print("   Recommended action:")
    print("   1. Run predictions on training set")
    print("   2. Manually review and complete masks")
    print("   3. Verify >90% coverage before retraining")
    print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print("Review generated files in dataset_analysis/")
print("  - distribution_analysis.png")
print("  - sample_images.png")
print("  - summary.json")
