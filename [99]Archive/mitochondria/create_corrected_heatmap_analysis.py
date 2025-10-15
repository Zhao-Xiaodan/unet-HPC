#!/usr/bin/env python3
"""
Create corrected hyperparameter heatmap analysis showing architecture-specific patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
from pathlib import Path

def load_hyperparameter_results(results_dir):
    """Load hyperparameter optimization results."""
    results = []
    exp_dirs = [d for d in Path(results_dir).iterdir() if d.is_dir() and d.name.startswith('exp_')]

    for exp_dir in sorted(exp_dirs):
        json_files = list(exp_dir.glob("*_results.json"))
        if json_files:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
                results.append(data)

    return pd.DataFrame(results)

def create_architecture_specific_heatmaps(hyperparameter_df, output_dir):
    """Create separate heatmaps for each architecture."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Hyperparameter Performance Heatmaps: Architecture-Specific Analysis',
                 fontsize=16, fontweight='bold')

    architectures = ['UNet', 'Attention_UNet', 'Attention_ResUNet']

    for i, arch in enumerate(architectures):
        ax = axes[i]

        # Filter data for this architecture
        arch_data = hyperparameter_df[hyperparameter_df['architecture'] == arch]

        # Create pivot table for this architecture only
        pivot_data = arch_data.pivot_table(
            values='best_val_jaccard',
            index='learning_rate',
            columns='batch_size',
            aggfunc='mean'
        )

        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax,
                   vmin=0.85, vmax=0.95, cbar_kws={'label': 'Jaccard Coefficient'})

        ax.set_title(f'{arch.replace("_", " ")}', fontweight='bold')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Learning Rate' if i == 0 else '')

        # Add sample size annotations
        for lr in pivot_data.index:
            for bs in pivot_data.columns:
                count = len(arch_data[(arch_data['learning_rate'] == lr) &
                                    (arch_data['batch_size'] == bs)])
                if count > 0:
                    ax.text(list(pivot_data.columns).index(bs) + 0.5,
                           list(pivot_data.index).index(lr) + 0.3,
                           f'n={count}', ha='center', va='center',
                           fontsize=8, alpha=0.7)

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir) / 'corrected_architecture_specific_heatmaps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved corrected heatmaps: {output_path}")
    return output_path

def analyze_architecture_differences(hyperparameter_df):
    """Analyze differences between architectures."""

    print("\\n🔍 ARCHITECTURE-SPECIFIC HYPERPARAMETER ANALYSIS:")
    print("="*60)

    for arch in ['UNet', 'Attention_UNet', 'Attention_ResUNet']:
        arch_data = hyperparameter_df[hyperparameter_df['architecture'] == arch]

        print(f"\\n📊 {arch}:")
        print(f"   Experiments: {len(arch_data)}")
        print(f"   Jaccard range: {arch_data['best_val_jaccard'].min():.3f} - {arch_data['best_val_jaccard'].max():.3f}")
        print(f"   Mean ± Std: {arch_data['best_val_jaccard'].mean():.3f} ± {arch_data['best_val_jaccard'].std():.3f}")

        # Find best configuration
        best_idx = arch_data['best_val_jaccard'].idxmax()
        best_config = arch_data.loc[best_idx]
        print(f"   Best config: LR={best_config['learning_rate']}, BS={best_config['batch_size']}, Jaccard={best_config['best_val_jaccard']:.3f}")

        # Show hyperparameter preferences
        best_lr = arch_data.groupby('learning_rate')['best_val_jaccard'].mean().idxmax()
        best_bs = arch_data.groupby('batch_size')['best_val_jaccard'].mean().idxmax()
        print(f"   Best avg LR: {best_lr}")
        print(f"   Best avg BS: {best_bs}")

def main():
    # Load data
    results_dir = "hyperparameter_optimization_20250927_101211"
    output_dir = "breakthrough_analysis_20250928"

    print("📊 CREATING CORRECTED ARCHITECTURE-SPECIFIC HEATMAP ANALYSIS")
    print("="*60)

    # Load hyperparameter results
    hyperparameter_df = load_hyperparameter_results(results_dir)

    # Create corrected heatmaps
    heatmap_path = create_architecture_specific_heatmaps(hyperparameter_df, output_dir)

    # Analyze architecture differences
    analyze_architecture_differences(hyperparameter_df)

    print(f"\\n✅ CORRECTED ANALYSIS COMPLETE!")
    print(f"📊 Architecture-specific heatmaps: {heatmap_path}")
    print(f"🎯 This shows the REAL hyperparameter preferences for each architecture")

if __name__ == "__main__":
    main()