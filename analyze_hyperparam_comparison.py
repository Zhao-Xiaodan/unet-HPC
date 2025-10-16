#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Search Analysis
Compares UNet, Attention UNet, and Attention ResUNet architectures
Date: October 16, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'unet': {
        'results_csv': './unet_hyperparam_20251015_224125/unet_results.csv',
        'dir': Path('./unet_hyperparam_20251015_224125'),
        'name': 'UNet',
        'color': '#3498db'  # Blue
    },
    'attention_unet': {
        'results_csv': './attention_unet_hyperparam_20251015_230149/attention_unet_results.csv',
        'dir': Path('./attention_unet_hyperparam_20251015_230149'),
        'name': 'Attention UNet',
        'color': '#e74c3c'  # Red
    },
    'attention_resunet': {
        'results_csv': './attention_resunet_hyperparam_20251015_235542/attention_resunet_results.csv',
        'dir': Path('./attention_resunet_hyperparam_20251015_235542'),
        'name': 'Attention ResUNet',
        'color': '#2ecc71'  # Green
    }
}

OUTPUT_DIR = Path('./hyperparam_comparison_report')
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

def load_all_results():
    """Load results from all three architectures."""
    results = {}

    for arch, config in CONFIG.items():
        df = pd.read_csv(config['results_csv'])
        df['architecture'] = config['name']
        df['arch_color'] = config['color']
        results[arch] = df
        print(f"✓ Loaded {arch}: {len(df)} experiments")

    # Combine all results
    df_all = pd.concat([results['unet'], results['attention_unet'], results['attention_resunet']],
                       ignore_index=True)

    return results, df_all

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def find_best_models(results):
    """Find best model for each architecture."""
    best_models = {}

    for arch, df in results.items():
        best_idx = df['best_val_iou'].idxmax()
        best_model = df.loc[best_idx]
        best_models[arch] = best_model

        print(f"\n{CONFIG[arch]['name']} Best Model:")
        print(f"  IoU: {best_model['best_val_iou']:.4f}")
        print(f"  Dice: {best_model['best_val_dice']:.4f}")
        print(f"  Filters: {best_model['n_filters']}")
        print(f"  Dropout: {best_model['dropout']}")
        print(f"  Learning Rate: {best_model['learning_rate']}")
        print(f"  Best Epoch: {best_model['best_epoch']}")

    return best_models

def compute_statistics(results, df_all):
    """Compute summary statistics for each architecture."""
    stats = []

    for arch, df in results.items():
        stats.append({
            'Architecture': CONFIG[arch]['name'],
            'Mean IoU': df['best_val_iou'].mean(),
            'Std IoU': df['best_val_iou'].std(),
            'Max IoU': df['best_val_iou'].max(),
            'Min IoU': df['best_val_iou'].min(),
            'Median IoU': df['best_val_iou'].median(),
            'Mean Dice': df['best_val_dice'].mean(),
            'Std Dice': df['best_val_dice'].std(),
            'Max Dice': df['best_val_dice'].max(),
        })

    df_stats = pd.DataFrame(stats)
    return df_stats

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_best_iou_comparison(df_all, best_models):
    """Bar plot comparing best IoU scores."""
    fig, ax = plt.subplots(figsize=(10, 6))

    arch_names = [CONFIG[arch]['name'] for arch in ['unet', 'attention_unet', 'attention_resunet']]
    best_ious = [best_models[arch]['best_val_iou'] for arch in ['unet', 'attention_unet', 'attention_resunet']]
    colors = [CONFIG[arch]['color'] for arch in ['unet', 'attention_unet', 'attention_resunet']]

    bars = ax.bar(arch_names, best_ious, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, iou in zip(bars, best_ious):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{iou:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Best Validation IoU', fontsize=12, fontweight='bold')
    ax.set_title('Best Model Comparison: Validation IoU', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(best_ious) * 1.15])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_best_iou_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig1_best_iou_comparison.png")

def plot_iou_distribution(df_all):
    """Box plot showing IoU distribution for each architecture."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create box plot
    positions = [0, 1, 2]
    box_data = [df_all[df_all['architecture'] == CONFIG[arch]['name']]['best_val_iou'].values
                for arch in ['unet', 'attention_unet', 'attention_resunet']]

    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    # Color boxes
    colors = [CONFIG[arch]['color'] for arch in ['unet', 'attention_unet', 'attention_resunet']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels([CONFIG[arch]['name'] for arch in ['unet', 'attention_unet', 'attention_resunet']])
    ax.set_ylabel('Best Validation IoU', fontsize=12, fontweight='bold')
    ax.set_title('Validation IoU Distribution Across Hyperparameters', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_iou_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig2_iou_distribution.png")

def plot_hyperparameter_heatmaps(results):
    """Heatmaps showing IoU across hyperparameter combinations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (arch, df) in enumerate(results.items()):
        # Create pivot table: n_filters × learning_rate, averaged over dropout
        pivot = df.pivot_table(values='best_val_iou',
                              index='n_filters',
                              columns='learning_rate',
                              aggfunc='mean')

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=axes[idx], cbar_kws={'label': 'IoU'},
                   vmin=0.2, vmax=0.55, linewidths=0.5)

        axes[idx].set_title(f'{CONFIG[arch]["name"]}\n(averaged over dropout)',
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Learning Rate', fontsize=11)
        axes[idx].set_ylabel('Number of Filters', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_hyperparameter_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig3_hyperparameter_heatmaps.png")

def plot_dropout_effect(df_all):
    """Line plot showing effect of dropout on performance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for arch in ['unet', 'attention_unet', 'attention_resunet']:
        df_arch = df_all[df_all['architecture'] == CONFIG[arch]['name']]

        # Group by dropout and compute mean IoU
        dropout_means = df_arch.groupby('dropout')['best_val_iou'].mean()
        dropout_stds = df_arch.groupby('dropout')['best_val_iou'].std()

        ax.plot(dropout_means.index, dropout_means.values,
               marker='o', linewidth=2, markersize=8,
               label=CONFIG[arch]['name'], color=CONFIG[arch]['color'])

        # Add error bars
        ax.fill_between(dropout_means.index,
                       dropout_means.values - dropout_stds.values,
                       dropout_means.values + dropout_stds.values,
                       alpha=0.2, color=CONFIG[arch]['color'])

    ax.set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Best Validation IoU', fontsize=12, fontweight='bold')
    ax.set_title('Effect of Dropout on Model Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.1, 0.2, 0.3])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_dropout_effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig4_dropout_effect.png")

def plot_learning_rate_effect(df_all):
    """Line plot showing effect of learning rate on performance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for arch in ['unet', 'attention_unet', 'attention_resunet']:
        df_arch = df_all[df_all['architecture'] == CONFIG[arch]['name']]

        # Group by learning rate and compute mean IoU
        lr_means = df_arch.groupby('learning_rate')['best_val_iou'].mean()
        lr_stds = df_arch.groupby('learning_rate')['best_val_iou'].std()

        ax.plot(lr_means.index, lr_means.values,
               marker='s', linewidth=2, markersize=8,
               label=CONFIG[arch]['name'], color=CONFIG[arch]['color'])

        # Add error bars
        ax.fill_between(lr_means.index,
                       lr_means.values - lr_stds.values,
                       lr_means.values + lr_stds.values,
                       alpha=0.2, color=CONFIG[arch]['color'])

    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Best Validation IoU', fontsize=12, fontweight='bold')
    ax.set_title('Effect of Learning Rate on Model Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.001, 0.003, 0.005])
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_learning_rate_effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig5_learning_rate_effect.png")

def plot_n_filters_effect(df_all):
    """Bar plot showing effect of number of filters."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(3)  # 3 filter values: 16, 32, 64
    width = 0.25

    for idx, arch in enumerate(['unet', 'attention_unet', 'attention_resunet']):
        df_arch = df_all[df_all['architecture'] == CONFIG[arch]['name']]

        # Group by n_filters and compute mean IoU
        filter_means = df_arch.groupby('n_filters')['best_val_iou'].mean()
        filter_stds = df_arch.groupby('n_filters')['best_val_iou'].std()

        offset = width * (idx - 1)
        ax.bar(x + offset, filter_means.values, width,
              label=CONFIG[arch]['name'], color=CONFIG[arch]['color'],
              alpha=0.7, yerr=filter_stds.values, capsize=5)

    ax.set_xlabel('Number of Filters (Initial Layer)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Best Validation IoU', fontsize=12, fontweight='bold')
    ax.set_title('Effect of Model Capacity on Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([16, 32, 64])
    ax.legend(fontsize=11, loc='best')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_n_filters_effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig6_n_filters_effect.png")

def plot_convergence_epoch(df_all):
    """Box plot showing convergence epochs (best_epoch)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = [0, 1, 2]
    box_data = [df_all[df_all['architecture'] == CONFIG[arch]['name']]['best_epoch'].values
                for arch in ['unet', 'attention_unet', 'attention_resunet']]

    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True,
                    medianprops=dict(color='red', linewidth=2))

    colors = [CONFIG[arch]['color'] for arch in ['unet', 'attention_unet', 'attention_resunet']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels([CONFIG[arch]['name'] for arch in ['unet', 'attention_unet', 'attention_resunet']])
    ax.set_ylabel('Best Epoch', fontsize=12, fontweight='bold')
    ax.set_title('Training Convergence Speed (Epoch of Best Validation IoU)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_convergence_epoch.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig7_convergence_epoch.png")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("HYPERPARAMETER SEARCH COMPARISON ANALYSIS")
    print("="*80)
    print()

    # Load data
    print("Loading results...")
    results, df_all = load_all_results()
    print()

    # Find best models
    print("="*80)
    print("BEST MODELS")
    print("="*80)
    best_models = find_best_models(results)
    print()

    # Compute statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    df_stats = compute_statistics(results, df_all)
    print(df_stats.to_string(index=False))
    df_stats.to_csv(OUTPUT_DIR / 'summary_statistics.csv', index=False)
    print(f"\n✓ Saved: summary_statistics.csv")
    print()

    # Generate visualizations
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    plot_best_iou_comparison(df_all, best_models)
    plot_iou_distribution(df_all)
    plot_hyperparameter_heatmaps(results)
    plot_dropout_effect(df_all)
    plot_learning_rate_effect(df_all)
    plot_n_filters_effect(df_all)
    plot_convergence_epoch(df_all)
    print()

    # Save best models info
    best_models_info = []
    for arch, model in best_models.items():
        best_models_info.append({
            'Architecture': CONFIG[arch]['name'],
            'Experiment': model['experiment_name'],
            'Best IoU': model['best_val_iou'],
            'Best Dice': model['best_val_dice'],
            'Best Epoch': model['best_epoch'],
            'n_filters': model['n_filters'],
            'dropout': model['dropout'],
            'learning_rate': model['learning_rate']
        })

    df_best = pd.DataFrame(best_models_info)
    df_best.to_csv(OUTPUT_DIR / 'best_models_summary.csv', index=False)
    print(f"✓ Saved: best_models_summary.csv")
    print()

    print("="*80)
    print(f"✓ Analysis complete! Results saved to: {OUTPUT_DIR}/")
    print("="*80)

if __name__ == '__main__':
    main()
