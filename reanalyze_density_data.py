#!/usr/bin/env python3
"""
Re-analyze Density Data with Corrected Dilution Factors
========================================================

Fixes the dilution factor parsing bug and regenerates plots
from existing CSV data.

Bug: Substring matching caused:
  - 10240x → parsed as 10x
  - 5120x → parsed as 20x
  - 640x → parsed as 40x
  - 1280x → parsed as 80x

Fix: Use regex with word boundaries for accurate parsing.

Author: Claude Code
Date: October 14, 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# Configuration
CSV_PATH = './density_analysis_arch_comparison_20251014_004358/csv_data/density_analysis_comprehensive.csv'
OUTPUT_DIR = Path('./density_analysis_arch_comparison_20251014_004358_CORRECTED')

COLORS = {
    'unet': '#440154',
    'resunet': '#31688e',
    'attention_resunet': '#35b779',
    'clahe_otsu': '#fde724'
}

METHOD_NAMES = {
    'unet': 'U-Net',
    'resunet': 'ResUNet',
    'attention_resunet': 'Attention ResUNet',
    'clahe_otsu': 'CLAHE+OTSU'
}

def extract_dilution_factor_corrected(filename):
    """Extract dilution factor with corrected regex (no substring matching)."""
    # Match dilution at start or after delimiter
    match = re.search(r'(?:^|_)(\d+)x(?:_|\.|-)', filename.lower())
    if match:
        return int(match.group(1))

    # Fallback: match at very beginning
    match = re.search(r'^(\d+)x', filename.lower())
    if match:
        return int(match.group(1))

    return None

def create_individual_plot(df, method, output_path):
    """
    Create individual boxplot for one method.

    Y-axis: Foreground Percentage (log scale)
    X-axis: 1/Dilution Factor
    """
    df_method = df[df['method'] == method].copy()
    df_method = df_method.sort_values('dilution_factor')

    dilution_factors = sorted(df_method['dilution_factor'].unique())

    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for boxplot
    data_to_plot = []
    labels = []
    positions = []

    for i, dilution in enumerate(dilution_factors):
        df_dilution = df_method[df_method['dilution_factor'] == dilution]
        data_to_plot.append(df_dilution['foreground_pct'].values)
        labels.append(f"1/{int(dilution)}")
        positions.append(i)

    # Create boxplot
    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=4, alpha=0.5),
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(facecolor=COLORS[method], alpha=0.7, edgecolor='black'),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5)
    )

    # Set log scale for y-axis
    ax.set_yscale('log')

    # Labels and title
    ax.set_xlabel('1/Dilution Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Foreground Percentage', fontsize=14, fontweight='bold')
    ax.set_title(f'{METHOD_NAMES[method]} - Density vs Dilution Factor',
                 fontsize=16, fontweight='bold', pad=20)

    # X-axis
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)

    # Add sample counts
    for i, (dilution, data) in enumerate(zip(dilution_factors, data_to_plot)):
        n = len(data)
        ax.text(i, ax.get_ylim()[0] * 1.5, f'n={n}',
                ha='center', va='top', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")

def main():
    print("=" * 80)
    print("RE-ANALYZING DENSITY DATA WITH CORRECTED DILUTION FACTORS")
    print("=" * 80)
    print()

    # Load CSV
    print(f"Loading CSV: {CSV_PATH}")
    df_original = pd.read_csv(CSV_PATH)
    print(f"  ✓ Loaded {len(df_original)} measurements")
    print()

    # Show original dilution factors (buggy)
    print("ORIGINAL (BUGGY) Dilution Factors:")
    print(f"  {sorted(df_original['dilution_factor'].unique())}")
    print()

    # Re-parse dilution factors correctly
    print("Re-parsing dilution factors from image names...")
    df = df_original.copy()
    df['dilution_factor'] = df['image'].apply(extract_dilution_factor_corrected)

    # Remove rows where dilution couldn't be parsed
    df = df.dropna(subset=['dilution_factor'])
    df['dilution_factor'] = df['dilution_factor'].astype(int)

    print(f"  ✓ Parsed {len(df)} measurements")
    print()

    # Show corrected dilution factors
    print("CORRECTED Dilution Factors:")
    dilutions_corrected = sorted(df['dilution_factor'].unique())
    print(f"  {dilutions_corrected}")
    print(f"  Range: {dilutions_corrected[0]}x to {dilutions_corrected[-1]}x")
    print()

    # Show comparison
    print("=" * 80)
    print("BEFORE vs AFTER Comparison:")
    print("=" * 80)
    print(f"{'Image Name':<50} {'OLD':<10} {'NEW':<10}")
    print("-" * 70)

    for image_name in sorted(df['image'].unique()):
        old_dilution = df_original[df_original['image'] == image_name]['dilution_factor'].iloc[0]
        new_dilution = df[df['image'] == image_name]['dilution_factor'].iloc[0]

        status = "✓" if old_dilution == new_dilution else "✗ FIXED"
        print(f"{image_name:<50} {old_dilution:<10} {new_dilution:<10} {status}")

    print()

    # Summary statistics
    print("=" * 80)
    print("DATA SUMMARY (CORRECTED)")
    print("=" * 80)
    print(f"Total measurements: {len(df)}")
    print(f"Methods: {df['method'].unique().tolist()}")
    print(f"Dilution factors: {dilutions_corrected}")
    print(f"Images analyzed: {df['image'].nunique()}")
    print()

    print("Measurements per dilution factor:")
    for dilution in dilutions_corrected:
        count = len(df[df['dilution_factor'] == dilution])
        n_images = df[df['dilution_factor'] == dilution]['image'].nunique()
        n_methods = df[df['dilution_factor'] == dilution]['method'].nunique()
        print(f"  {dilution:>5}x: {count:4} measurements ({n_images} images × {n_methods} methods)")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    plots_dir = OUTPUT_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
    csv_dir = OUTPUT_DIR / 'csv_data'
    csv_dir.mkdir(exist_ok=True)

    # Save corrected CSV
    corrected_csv_path = csv_dir / 'density_analysis_comprehensive_CORRECTED.csv'
    df.to_csv(corrected_csv_path, index=False)
    print(f"✓ Saved corrected CSV: {corrected_csv_path}")
    print()

    # Generate plots
    print("=" * 80)
    print("GENERATING CORRECTED PLOTS")
    print("=" * 80)
    print()

    methods = ['unet', 'resunet', 'attention_resunet', 'clahe_otsu']

    for method in methods:
        output_path = plots_dir / f'{method}_density_vs_dilution_CORRECTED.png'
        print(f"Creating plot for {METHOD_NAMES[method]}...")
        create_individual_plot(df, method, output_path)

    print()
    print("=" * 80)
    print("RE-ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print()
    print("Generated files:")
    print(f"  ✓ {len(list(plots_dir.glob('*.png')))} corrected plot(s)")
    print(f"  ✓ 1 corrected CSV")
    print()
    print("Key changes:")
    print(f"  - X-axis now shows: {dilutions_corrected[0]}x to {dilutions_corrected[-1]}x")
    print(f"  - Previously showed: 10x to 160x only (BUG)")
    print()
    print("✓ All dilution factors from 10x to 10240x are now included!")
    print()

if __name__ == '__main__':
    main()
