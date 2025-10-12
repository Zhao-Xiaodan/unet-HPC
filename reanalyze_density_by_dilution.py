#!/usr/bin/env python3
"""
Re-analyze Density Data by Dilution Factor
===========================================
Extracts dilution factors from image names and creates boxplots
grouped by dilution factor with architectures side-by-side.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# Configuration
RESULTS_DIR = './prediction_analysis_20251012_074415'
OUTPUT_DIR = './density_analysis_dilution_factors'
CSV_FILE = f'{RESULTS_DIR}/summary/density_analysis_summary.csv'

# Dilution factor mapping
DILUTION_PATTERNS = {
    '10x': 10,
    '20x': 20,
    '80x': 80,
    '160x': 160,
    '320x': 320,
    '640x': 640,
    '1280x': 1280,
    '2560x': 2560,
    '5120x': 5120,
    '10240x': 10240
}


def extract_dilution_factor(image_name):
    """
    Extract dilution factor from image name

    Examples:
        '10x_2025-05-15_02-05-00' -> 10
        '80x_1_2025-05-22_14-48-00_003' -> 80
        '10240x_2560x_2025-05-16_00-59-00_002' -> 10240

    Returns:
        Dilution factor as integer, or None if not found
    """
    # Try to match known dilution patterns
    for pattern, value in DILUTION_PATTERNS.items():
        if image_name.startswith(pattern):
            return value

    # Try to extract number followed by 'x' at start of string
    match = re.match(r'^(\d+)x', image_name)
    if match:
        return int(match.group(1))

    return None


def create_output_dir():
    """Create output directory"""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    print(f"✓ Output directory: {output_path}")
    return output_path


def load_and_process_data(csv_path):
    """
    Load CSV and add dilution factor column

    Returns:
        DataFrame with dilution_factor column added
    """
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"✓ Loaded {len(df)} rows")

    # Extract dilution factors
    df['dilution_factor'] = df['image'].apply(extract_dilution_factor)

    # Remove rows without dilution factor
    df_clean = df[df['dilution_factor'].notna()].copy()

    print(f"✓ Found {len(df_clean)} rows with dilution factors")
    print(f"  Unique dilution factors: {sorted(df_clean['dilution_factor'].unique())}")
    print(f"  Architectures: {df_clean['architecture'].unique().tolist()}")

    return df_clean


def create_boxplot_by_dilution(df, output_dir, metric='mean_density'):
    """
    Create boxplot with dilution factors on x-axis, grouped by architecture

    Args:
        df: DataFrame with dilution_factor and architecture columns
        output_dir: Path to save plot
        metric: Column name for y-axis (default: 'mean_density')
    """
    print(f"\nCreating boxplot for {metric}...")

    # Sort by dilution factor
    df_sorted = df.sort_values('dilution_factor')

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Define architecture order and colors
    arch_order = ['clahe_otsu', 'unet', 'resunet', 'attention_resunet']
    colors = {
        'clahe_otsu': '#2ecc71',      # Green (reference)
        'unet': '#3498db',             # Blue
        'resunet': '#e74c3c',          # Red
        'attention_resunet': '#f39c12' # Orange
    }

    # Get unique dilution factors
    dilution_factors = sorted(df_sorted['dilution_factor'].unique())

    # Prepare data for grouped boxplot
    positions = []
    data_to_plot = []
    labels = []
    box_colors = []

    n_archs = len(arch_order)
    box_width = 0.8
    group_width = n_archs * box_width
    group_spacing = 1.5

    for i, dilution in enumerate(dilution_factors):
        df_dilution = df_sorted[df_sorted['dilution_factor'] == dilution]

        # Calculate center position for this dilution group
        group_center = i * (group_width + group_spacing)

        for j, arch in enumerate(arch_order):
            df_arch = df_dilution[df_dilution['architecture'] == arch]

            if len(df_arch) > 0:
                # Position for this architecture within the group
                pos = group_center + (j - n_archs/2 + 0.5) * box_width

                positions.append(pos)
                data_to_plot.append(df_arch[metric].values)
                labels.append(f"{int(dilution)}x\n{arch.replace('_', ' ')}")
                box_colors.append(colors[arch])

    # Create boxplot
    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=box_width * 0.7,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(linewidth=1.5),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5)
    )

    # Color boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add scatter points
    for pos, data, color in zip(positions, data_to_plot, box_colors):
        # Add jitter
        x = np.random.normal(pos, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.4, s=30, color=color, zorder=3)

    # Set x-axis labels (only dilution factors, centered on groups)
    x_tick_positions = []
    x_tick_labels = []
    for i, dilution in enumerate(dilution_factors):
        group_center = i * (group_width + group_spacing)
        x_tick_positions.append(group_center)
        x_tick_labels.append(f'{int(dilution)}x')

    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, fontsize=12, fontweight='bold')

    # Labels and title
    ax.set_xlabel('Dilution Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Particle Density (fraction)', fontsize=14, fontweight='bold')
    ax.set_title('Particle Density by Dilution Factor and Architecture',
                 fontsize=16, fontweight='bold', pad=20)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['clahe_otsu'], alpha=0.7, label='CLAHE+OTSU (Reference)'),
        Patch(facecolor=colors['unet'], alpha=0.7, label='U-Net'),
        Patch(facecolor=colors['resunet'], alpha=0.7, label='ResU-Net'),
        Patch(facecolor=colors['attention_resunet'], alpha=0.7, label='Attention ResU-Net')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    # Adjust y-axis
    y_max = df_sorted[metric].max()
    if y_max > 0.9:  # If ResU-Net is predicting all white
        ax.set_ylim(-0.05, 1.05)
        # Add warning annotation
        ax.text(0.5, 0.95, '⚠ ResU-Net predicting ~100% foreground (likely model issue)',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                fontsize=10, fontweight='bold')
    else:
        ax.set_ylim(0, y_max * 1.1)

    plt.tight_layout()

    # Save
    output_path = output_dir / f'density_by_dilution_{metric}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def create_separate_plots(df, output_dir):
    """
    Create separate plots:
    1. CLAHE+OTSU only (reference)
    2. Deep learning models only (for comparison)
    """
    print("\nCreating separate comparison plots...")

    # Plot 1: CLAHE+OTSU reference only
    df_ref = df[df['architecture'] == 'clahe_otsu'].copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    dilution_factors = sorted(df_ref['dilution_factor'].unique())

    # Boxplot
    data_to_plot = [df_ref[df_ref['dilution_factor'] == d]['mean_density'].values
                    for d in dilution_factors]

    bp = ax.boxplot(data_to_plot, labels=[f'{int(d)}x' for d in dilution_factors],
                    patch_artist=True, showfliers=True,
                    boxprops=dict(facecolor='#2ecc71', alpha=0.7, linewidth=1.5),
                    medianprops=dict(color='black', linewidth=2))

    # Add scatter
    for i, (d, data) in enumerate(zip(dilution_factors, data_to_plot)):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.4, s=40, color='#27ae60', zorder=3)

    ax.set_xlabel('Dilution Factor', fontsize=13, fontweight='bold')
    ax.set_ylabel('Particle Density (fraction)', fontsize=13, fontweight='bold')
    ax.set_title('Particle Density by Dilution Factor (CLAHE+OTSU Reference Method)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'density_clahe_otsu_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: density_clahe_otsu_only.png")

    # Plot 2: Deep learning models comparison (excluding problematic ResU-Net)
    df_dl = df[df['architecture'].isin(['unet', 'attention_resunet'])].copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    arch_order = ['unet', 'attention_resunet']
    colors = {'unet': '#3498db', 'attention_resunet': '#f39c12'}

    positions = []
    data_to_plot = []
    box_colors = []

    n_archs = len(arch_order)
    box_width = 0.4

    for i, dilution in enumerate(dilution_factors):
        for j, arch in enumerate(arch_order):
            df_subset = df_dl[(df_dl['dilution_factor'] == dilution) &
                             (df_dl['architecture'] == arch)]

            if len(df_subset) > 0:
                pos = i * (n_archs * box_width + 0.5) + j * box_width
                positions.append(pos)
                data_to_plot.append(df_subset['mean_density'].values)
                box_colors.append(colors[arch])

    bp = ax.boxplot(data_to_plot, positions=positions, widths=box_width * 0.8,
                    patch_artist=True, showfliers=True)

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # X-axis labels
    x_ticks = [(i * (n_archs * box_width + 0.5) + box_width/2) for i in range(len(dilution_factors))]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{int(d)}x' for d in dilution_factors], fontsize=11)

    ax.set_xlabel('Dilution Factor', fontsize=13, fontweight='bold')
    ax.set_ylabel('Particle Density (fraction)', fontsize=13, fontweight='bold')
    ax.set_title('Deep Learning Model Comparison (U-Net vs Attention ResU-Net)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['unet'], alpha=0.7, label='U-Net'),
        Patch(facecolor=colors['attention_resunet'], alpha=0.7, label='Attention ResU-Net')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'density_dl_models_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: density_dl_models_only.png")


def print_summary_statistics(df):
    """Print summary statistics grouped by architecture and dilution"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY ARCHITECTURE")
    print("="*80)

    for arch in ['clahe_otsu', 'unet', 'resunet', 'attention_resunet']:
        df_arch = df[df['architecture'] == arch]
        if len(df_arch) == 0:
            continue

        print(f"\n{arch.upper().replace('_', ' ')}")
        print(f"  Overall mean density: {df_arch['mean_density'].mean():.4f} ± {df_arch['mean_density'].std():.4f}")
        print(f"  Range: [{df_arch['mean_density'].min():.4f}, {df_arch['mean_density'].max():.4f}]")

        # Check for issues
        if df_arch['mean_density'].mean() > 0.9:
            print(f"  ⚠ WARNING: Predicting ~{df_arch['mean_density'].mean()*100:.1f}% foreground!")
            print(f"             This suggests model is predicting almost all pixels as particles.")
        elif df_arch['mean_density'].mean() < 0.01:
            print(f"  ⚠ WARNING: Predicting only {df_arch['mean_density'].mean()*100:.2f}% foreground!")
            print(f"             This suggests model is predicting almost no particles.")

    print("\n" + "="*80)
    print("PARTICLE DENSITY BY DILUTION FACTOR (CLAHE+OTSU Reference)")
    print("="*80)

    df_ref = df[df['architecture'] == 'clahe_otsu'].sort_values('dilution_factor')

    print(f"\n{'Dilution':>10s} {'Mean Density':>15s} {'Std Dev':>15s} {'Range':>25s}")
    print("-" * 70)

    for dilution in sorted(df_ref['dilution_factor'].unique()):
        df_d = df_ref[df_ref['dilution_factor'] == dilution]
        mean = df_d['mean_density'].mean()
        std = df_d['mean_density'].std()
        min_val = df_d['mean_density'].min()
        max_val = df_d['mean_density'].max()
        print(f"{int(dilution):>8d}x {mean:>15.4f} {std:>15.4f} [{min_val:>8.4f}, {max_val:>8.4f}]")


def main():
    print("="*80)
    print("RE-ANALYSIS: PARTICLE DENSITY BY DILUTION FACTOR")
    print("="*80)

    # Create output directory
    output_dir = create_output_dir()

    # Load and process data
    df = load_and_process_data(CSV_FILE)

    if len(df) == 0:
        print("✗ No data with dilution factors found!")
        return

    # Print summary statistics
    print_summary_statistics(df)

    # Create plots
    create_boxplot_by_dilution(df, output_dir, metric='mean_density')
    create_separate_plots(df, output_dir)

    # Save processed data
    output_csv = output_dir / 'density_with_dilution_factors.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved processed data: {output_csv}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput files in: {output_dir}/")
    print("  - density_by_dilution_mean_density.png (all architectures)")
    print("  - density_clahe_otsu_only.png (reference method)")
    print("  - density_dl_models_only.png (U-Net vs Attention ResU-Net)")
    print("  - density_with_dilution_factors.csv (processed data)")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
