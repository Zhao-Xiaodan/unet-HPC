#!/usr/bin/env python3
"""
PyTorch Density Analysis with 4-Panel Tile Visualizations
==========================================================

Analyzes predictions from three PyTorch architectures (UNet, Attention UNet,
Attention ResUNet) and generates density plots with 4-panel representative tiles.

Features:
- Density boxplots (full range and low dilution range)
- Threshold 0.5 only (simplified from previous analysis)
- 4-panel tiles: Original | UNet Pred | Attention UNet Pred | Attention ResUNet Pred
- 5 representative tiles per dilution factor

Usage:
    python density_analysis_pytorch_comparison.py \\
        --predictions <pred_dir> \\
        --test_images <test_img_dir> \\
        --output <output_dir>

Author: Claude Code
Date: October 22, 2025
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import argparse
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dilution factors (low density → high density)
DILUTION_ORDER = [10240, 5120, 2560, 1280, 640, 320, 160, 80, 20, 10]
DILUTION_LABELS = ['1/10240x', '1/5120x', '1/2560x', '1/1280x', '1/640x',
                   '1/320x', '1/160x', '1/80x', '1/20x', '1/10x']

# Low dilution range (1/10240 to 1/80)
DILUTION_ORDER_LOW = [10240, 5120, 2560, 1280, 640, 320, 160, 80]
DILUTION_LABELS_LOW = ['1/10240x', '1/5120x', '1/2560x', '1/1280x',
                       '1/640x', '1/320x', '1/160x', '1/80x']

# Dilution patterns for filename extraction
DILUTION_PATTERNS = {
    '10240x': 10240, '5120x': 5120, '2560x': 2560, '1280x': 1280,
    '640x': 640, '320x': 320, '160x': 160, '80x': 80, '20x': 20, '10x': 10
}

THRESHOLD = 0.5  # Single threshold for density calculation
TILE_SIZE = 512

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_dilution_from_filename(filename):
    """Extract dilution factor from filename"""
    filename_lower = filename.lower()
    for pattern, dilution in DILUTION_PATTERNS.items():
        if pattern in filename_lower:
            return dilution
    return 1

def load_image(image_path):
    """Load image (grayscale)"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img

def load_prediction(pred_path):
    """Load prediction mask (0-255 grayscale)"""
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    if pred is None:
        raise ValueError(f"Could not load prediction: {pred_path}")
    return pred.astype(np.float32) / 255.0  # Normalize to [0, 1]

def extract_tiles(image, tile_size=512):
    """Extract non-overlapping tiles"""
    h, w = image.shape[:2]
    tiles = []
    positions = []

    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((y, x))

    return tiles, positions

def calculate_density(mask, threshold=0.5):
    """Calculate bead density as percentage of pixels above threshold"""
    binary_mask = (mask > threshold).astype(np.float32)
    density = np.mean(binary_mask) * 100  # Percentage
    return density

# ============================================================================
# TILE-LEVEL DENSITY ANALYSIS
# ============================================================================

def analyze_predictions(pred_dir, test_image_dir, threshold=0.5):
    """
    Analyze predictions and calculate tile-level densities.

    Returns:
        - tile_results: List of dicts with tile-level data
        - image_summary: List of dicts with image-level summary
    """
    pred_dir = Path(pred_dir)
    test_image_dir = Path(test_image_dir)

    # Get architecture subdirectories
    arch_dirs = {
        'unet': pred_dir / 'unet',
        'attention_unet': pred_dir / 'attention_unet',
        'attention_resunet': pred_dir / 'attention_resunet'
    }

    # Verify all architecture dirs exist
    for arch, dir_path in arch_dirs.items():
        if not dir_path.exists():
            raise FileNotFoundError(f"Architecture directory not found: {dir_path}")

    # Get test images
    test_images = sorted(test_image_dir.glob('*.tif'))

    tile_results = []
    image_summary = []

    print(f"\nAnalyzing {len(test_images)} images...")
    for img_path in tqdm(test_images, desc="Processing images"):
        # Extract dilution factor
        dilution = extract_dilution_from_filename(img_path.name)

        # Load original image
        original = load_image(img_path)

        # Extract tiles from original
        orig_tiles, positions = extract_tiles(original, TILE_SIZE)

        # Load predictions for each architecture
        predictions = {}
        for arch, dir_path in arch_dirs.items():
            pred_path = dir_path / f"{img_path.stem}_pred.png"
            if not pred_path.exists():
                raise FileNotFoundError(f"Prediction not found: {pred_path}")
            predictions[arch] = load_prediction(pred_path)

        # Extract tiles from predictions
        pred_tiles = {}
        for arch, pred_img in predictions.items():
            tiles, _ = extract_tiles(pred_img, TILE_SIZE)
            pred_tiles[arch] = tiles

        # Calculate densities for each tile
        image_densities = {arch: [] for arch in arch_dirs.keys()}

        for tile_idx in range(len(orig_tiles)):
            tile_data = {
                'image': img_path.stem,
                'dilution': dilution,
                'tile_idx': tile_idx,
                'position_y': positions[tile_idx][0],
                'position_x': positions[tile_idx][1]
            }

            # Calculate density for each architecture
            for arch in arch_dirs.keys():
                density = calculate_density(pred_tiles[arch][tile_idx], threshold)
                tile_data[f'{arch}_density'] = density
                image_densities[arch].append(density)

            tile_results.append(tile_data)

        # Image-level summary
        summary = {
            'image': img_path.stem,
            'dilution': dilution,
            'n_tiles': len(orig_tiles)
        }

        for arch in arch_dirs.keys():
            summary[f'{arch}_mean_density'] = np.mean(image_densities[arch])
            summary[f'{arch}_std_density'] = np.std(image_densities[arch])
            summary[f'{arch}_median_density'] = np.median(image_densities[arch])

        image_summary.append(summary)

    return tile_results, image_summary

# ============================================================================
# VISUALIZATION: DENSITY BOXPLOTS
# ============================================================================

def create_density_boxplot(df_tiles, dilution_order, dilution_labels,
                           output_path, title_suffix="", use_log_scale=True):
    """Create density boxplot for specified dilution range"""

    # Filter to specified dilutions
    df_plot = df_tiles[df_tiles['dilution'].isin(dilution_order)].copy()

    # Create categorical dilution column for proper ordering
    df_plot['dilution_cat'] = pd.Categorical(
        df_plot['dilution'],
        categories=dilution_order,
        ordered=True
    )

    # Melt dataframe to long format
    df_long = pd.melt(
        df_plot,
        id_vars=['dilution_cat'],
        value_vars=['unet_density', 'attention_unet_density', 'attention_resunet_density'],
        var_name='Architecture',
        value_name='Density'
    )

    # Rename architectures for display
    arch_rename = {
        'unet_density': 'UNet',
        'attention_unet_density': 'Attention UNet',
        'attention_resunet_density': 'Attention ResUNet'
    }
    df_long['Architecture'] = df_long['Architecture'].map(arch_rename)

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    sns.boxplot(
        data=df_long,
        x='dilution_cat',
        y='Density',
        hue='Architecture',
        ax=ax,
        palette='Set2',
        showfliers=True
    )

    ax.set_xlabel('Dilution Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Bead Density (%) - Log Scale' if use_log_scale else 'Bead Density (%)',
                  fontsize=14, fontweight='bold')
    ax.set_title(f'Bead Density vs Dilution Factor (Threshold={THRESHOLD}) - {title_suffix}',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticklabels(dilution_labels, rotation=45, ha='right')
    ax.legend(title='Architecture', fontsize=11, title_fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Apply log scale if requested
    if use_log_scale:
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Saved: {output_path}")


def create_individual_model_boxplots(df_tiles, dilution_order, dilution_labels,
                                      output_dir, title_suffix="", use_log_scale=True):
    """
    Create separate boxplots for each individual model.
    This provides clearer per-architecture visualization.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Filter to specified dilutions
    df_plot = df_tiles[df_tiles['dilution'].isin(dilution_order)].copy()

    # Create categorical dilution column for proper ordering
    df_plot['dilution_cat'] = pd.Categorical(
        df_plot['dilution'],
        categories=dilution_order,
        ordered=True
    )

    # Architecture configurations
    architectures = [
        ('unet_density', 'UNet', 'tab:blue'),
        ('attention_unet_density', 'Attention UNet', 'tab:orange'),
        ('attention_resunet_density', 'Attention ResUNet', 'tab:green')
    ]

    for density_col, arch_name, color in architectures:
        fig, ax = plt.subplots(figsize=(16, 8))

        sns.boxplot(
            data=df_plot,
            x='dilution_cat',
            y=density_col,
            ax=ax,
            color=color,
            showfliers=True
        )

        ax.set_xlabel('Dilution Factor', fontsize=14, fontweight='bold')
        ax.set_ylabel('Bead Density (%) - Log Scale' if use_log_scale else 'Bead Density (%)',
                      fontsize=14, fontweight='bold')
        ax.set_title(f'{arch_name} - Bead Density vs Dilution Factor (Threshold={THRESHOLD}) - {title_suffix}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticklabels(dilution_labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # Apply log scale if requested
        if use_log_scale:
            ax.set_yscale('log')

        plt.tight_layout()

        # Save with architecture-specific filename
        arch_filename = arch_name.lower().replace(' ', '_')
        range_suffix = "full_range" if len(dilution_order) > 8 else "low_dilution_range"
        output_path = output_dir / f'density_boxplot_{arch_filename}_{range_suffix}_threshold_{THRESHOLD}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"  Saved: {output_path}")


# ============================================================================
# VISUALIZATION: 4-PANEL REPRESENTATIVE TILES
# ============================================================================

def create_4panel_representative_tiles(df_tiles, predictions_dir, test_images_dir, output_dir):
    """
    Create 4-panel tiles showing:
    - Column 1: Original image (512x512)
    - Column 2: Inverted UNet prediction (black background, white beads)
    - Column 3: Inverted Attention UNet prediction (black background, white beads)
    - Column 4: Inverted Attention ResUNet prediction (black background, white beads)

    IMPORTANT: Predictions are INVERTED to make predicted masks black (background)
    easier to compare with original images where beads appear as dark spots.

    Layout: 5 rows × 4 columns per dilution factor (5 representative tiles)
    Selection: Tiles are selected to represent density distribution (min, 25%, median, 75%, max)
    """
    predictions_dir = Path(predictions_dir)
    test_images_dir = Path(test_images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\nCreating 4-panel representative tiles (5 rows × 4 columns)...")

    for dilution in tqdm(DILUTION_ORDER, desc="Processing dilutions"):
        # Filter tiles for this dilution
        dilution_tiles = df_tiles[df_tiles['dilution'] == dilution]

        if len(dilution_tiles) == 0:
            print(f"  Warning: No tiles for {dilution}x")
            continue

        # Select 5 representative tiles (varied density distribution)
        n_tiles = min(5, len(dilution_tiles))

        # Sort by UNet density and select percentiles for representative spread
        sorted_tiles = dilution_tiles.sort_values('unet_density')
        n_total = len(sorted_tiles)

        # Select 5 tiles: min, 25th percentile, median, 75th percentile, max
        indices = [
            0,                      # Min density
            n_total // 4,           # 25th percentile
            n_total // 2,           # Median
            3 * n_total // 4,       # 75th percentile
            n_total - 1,            # Max density
        ]
        selected_tiles = sorted_tiles.iloc[indices[:n_tiles]]

        # Create figure with 5 rows × 4 columns
        fig, axes = plt.subplots(n_tiles, 4, figsize=(16, 4*n_tiles))

        if n_tiles == 1:
            axes = axes.reshape(1, -1)

        for row_idx, (_, tile_info) in enumerate(selected_tiles.iterrows()):
            image_name = tile_info['image']
            tile_idx = int(tile_info['tile_idx'])
            pos_y = int(tile_info['position_y'])
            pos_x = int(tile_info['position_x'])

            # Load original image
            orig_path = list(test_images_dir.glob(f"{image_name}.tif"))[0]
            orig_full = load_image(orig_path)
            orig_tile = orig_full[pos_y:pos_y+TILE_SIZE, pos_x:pos_x+TILE_SIZE]

            # Load predictions for each architecture
            pred_tiles = {}
            for arch in ['unet', 'attention_unet', 'attention_resunet']:
                pred_path = predictions_dir / arch / f"{image_name}_pred.png"
                pred_full = load_prediction(pred_path)
                pred_tile = pred_full[pos_y:pos_y+TILE_SIZE, pos_x:pos_x+TILE_SIZE]
                # CRITICAL: Invert prediction (0→1, 1→0)
                # This makes predicted background BLACK (easier to compare with originals)
                pred_tiles[arch] = 1.0 - pred_tile

            # Plot panels
            # Column 0: Original image
            axes[row_idx, 0].imshow(orig_tile, cmap='gray')
            axes[row_idx, 0].set_title(f'Original Image\nTile {tile_idx} @ ({pos_y},{pos_x})',
                                       fontsize=10, fontweight='bold')
            axes[row_idx, 0].axis('off')

            # Column 1: Inverted UNet prediction
            axes[row_idx, 1].imshow(pred_tiles['unet'], cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 1].set_title(f'Inverted UNet\n(Black=BG, White=Beads)\nDensity: {tile_info["unet_density"]:.2f}%',
                                      fontsize=10, fontweight='bold')
            axes[row_idx, 1].axis('off')

            # Column 2: Inverted Attention UNet prediction
            axes[row_idx, 2].imshow(pred_tiles['attention_unet'], cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 2].set_title(f'Inverted Attention UNet\n(Black=BG, White=Beads)\nDensity: {tile_info["attention_unet_density"]:.2f}%',
                                      fontsize=10, fontweight='bold')
            axes[row_idx, 2].axis('off')

            # Column 3: Inverted Attention ResUNet prediction
            axes[row_idx, 3].imshow(pred_tiles['attention_resunet'], cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 3].set_title(f'Inverted Attention ResUNet\n(Black=BG, White=Beads)\nDensity: {tile_info["attention_resunet_density"]:.2f}%',
                                      fontsize=10, fontweight='bold')
            axes[row_idx, 3].axis('off')

        # Overall title
        dilution_label = f'1/{dilution}x'
        fig.suptitle(f'Representative Tiles - Dilution {dilution_label} (5 rows × 4 columns, Threshold={THRESHOLD})',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()
        output_path = output_dir / f'tiles_4panel_{dilution}x.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    print(f"  ✓ Saved 4-panel tiles (5 rows × 4 columns) to: {output_dir}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PyTorch Density Analysis')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Directory containing predictions from all architectures')
    parser.add_argument('--test_images', type=str, required=True,
                       help='Directory containing original test images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for analysis results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PYTORCH DENSITY ANALYSIS - 4-PANEL COMPARISON")
    print("="*80)
    print(f"Predictions: {args.predictions}")
    print(f"Test images: {args.test_images}")
    print(f"Output: {args.output}")
    print(f"Threshold: {THRESHOLD}")
    print("="*80)

    # Step 1: Analyze predictions
    print("\n" + "="*80)
    print("STEP 1: ANALYZING PREDICTIONS")
    print("="*80)

    tile_results, image_summary = analyze_predictions(
        args.predictions,
        args.test_images,
        threshold=THRESHOLD
    )

    # Save results
    df_tiles = pd.DataFrame(tile_results)
    df_summary = pd.DataFrame(image_summary)

    df_tiles.to_csv(output_dir / 'density_results_tile_level.csv', index=False)
    df_summary.to_csv(output_dir / 'density_results_image_summary.csv', index=False)

    print(f"\n✓ Analyzed {len(df_summary)} images, {len(df_tiles)} tiles")
    print(f"  Saved: {output_dir / 'density_results_tile_level.csv'}")
    print(f"  Saved: {output_dir / 'density_results_image_summary.csv'}")

    # Step 2: Create density boxplots
    print("\n" + "="*80)
    print("STEP 2: CREATING DENSITY BOXPLOTS")
    print("="*80)

    # Full range boxplot (combined)
    print("\n2a. Combined Architecture Boxplots (with log scale):")
    create_density_boxplot(
        df_tiles,
        DILUTION_ORDER,
        DILUTION_LABELS,
        output_dir / f'density_boxplot_full_range__threshold_{THRESHOLD}.png',
        title_suffix="Full Range",
        use_log_scale=True
    )

    # Low dilution range boxplot (combined)
    create_density_boxplot(
        df_tiles,
        DILUTION_ORDER_LOW,
        DILUTION_LABELS_LOW,
        output_dir / f'density_boxplot_low_dilution_range__threshold_{THRESHOLD}.png',
        title_suffix="Low Dilution Range",
        use_log_scale=True
    )

    # Individual model boxplots - Full range
    print("\n2b. Individual Model Boxplots - Full Range (with log scale):")
    create_individual_model_boxplots(
        df_tiles,
        DILUTION_ORDER,
        DILUTION_LABELS,
        output_dir,
        title_suffix="Full Range",
        use_log_scale=True
    )

    # Individual model boxplots - Low dilution range
    print("\n2c. Individual Model Boxplots - Low Dilution Range (with log scale):")
    create_individual_model_boxplots(
        df_tiles,
        DILUTION_ORDER_LOW,
        DILUTION_LABELS_LOW,
        output_dir,
        title_suffix="Low Dilution Range",
        use_log_scale=True
    )

    # Step 3: Create 4-panel representative tiles
    print("\n" + "="*80)
    print("STEP 3: CREATING 4-PANEL REPRESENTATIVE TILES")
    print("="*80)

    tiles_output_dir = output_dir / 'representative_tiles_4panel'
    create_4panel_representative_tiles(
        df_tiles,
        args.predictions,
        args.test_images,
        tiles_output_dir
    )

    # Save experiment metadata
    metadata = {
        'predictions_dir': str(args.predictions),
        'test_images_dir': str(args.test_images),
        'output_dir': str(args.output),
        'threshold': THRESHOLD,
        'n_images': len(df_summary),
        'n_tiles': len(df_tiles),
        'dilution_factors': DILUTION_ORDER,
        'architectures': ['unet', 'attention_unet', 'attention_resunet'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(output_dir / 'EXPERIMENT_INFO.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for arch in ['unet', 'attention_unet', 'attention_resunet']:
        mean_col = f'{arch}_mean_density'
        overall_mean = df_summary[mean_col].mean()
        overall_std = df_summary[mean_col].std()

        arch_display = arch.replace('_', ' ').title()
        print(f"\n{arch_display}:")
        print(f"  Overall mean density: {overall_mean:.2f}% ± {overall_std:.2f}%")

        # By dilution
        print(f"  By dilution:")
        for dilution in DILUTION_ORDER:
            dilution_data = df_summary[df_summary['dilution'] == dilution]
            if len(dilution_data) > 0:
                mean_density = dilution_data[mean_col].mean()
                print(f"    1/{dilution}x: {mean_density:.2f}%")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - density_results_tile_level.csv")
    print("  - density_results_image_summary.csv")
    print(f"\n  Combined Boxplots (log scale):")
    print(f"    - density_boxplot_full_range__threshold_{THRESHOLD}.png")
    print(f"    - density_boxplot_low_dilution_range__threshold_{THRESHOLD}.png")
    print(f"\n  Individual Model Boxplots (log scale):")
    print(f"    - density_boxplot_unet_full_range_threshold_{THRESHOLD}.png")
    print(f"    - density_boxplot_attention_unet_full_range_threshold_{THRESHOLD}.png")
    print(f"    - density_boxplot_attention_resunet_full_range_threshold_{THRESHOLD}.png")
    print(f"    - density_boxplot_unet_low_dilution_range_threshold_{THRESHOLD}.png")
    print(f"    - density_boxplot_attention_unet_low_dilution_range_threshold_{THRESHOLD}.png")
    print(f"    - density_boxplot_attention_resunet_low_dilution_range_threshold_{THRESHOLD}.png")
    print(f"\n  Representative Tiles (5 rows × 4 columns, inverted predictions):")
    print("    - representative_tiles_4panel/ (10 images)")
    print("\n  - EXPERIMENT_INFO.json")

if __name__ == "__main__":
    main()
