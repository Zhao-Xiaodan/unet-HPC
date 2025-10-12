#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Search Analysis
============================================
Analyze results from hyperparam_comprehensive_20251012_005054
Generate visualizations and comprehensive report.
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

# Results directory
RESULTS_DIR = Path('hyperparam_comprehensive_20251012_005054')
OUTPUT_DIR = RESULTS_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

def load_results():
    """Load search results"""
    results_path = RESULTS_DIR / 'search_results_final.csv'
    df = pd.read_csv(results_path)
    return df

def load_training_history(arch, bs, dr, loss):
    """Load training history for specific configuration"""
    history_path = RESULTS_DIR / f'history_{arch}_bs{bs}_dr{dr}_{loss}.csv'
    if history_path.exists():
        return pd.read_csv(history_path)
    return None

def plot_top_configurations(df):
    """Plot top 10 configurations comparison"""
    fig, ax = plt.subplots(figsize=(14, 8))

    top10 = df.head(10).copy()
    top10['config'] = top10.apply(
        lambda x: f"{x['architecture']}\nBS={int(x['batch_size'])}\n{x['loss_function']}",
        axis=1
    )

    x = np.arange(len(top10))
    width = 0.35

    bars1 = ax.bar(x - width/2, top10['best_val_jacard'], width,
                   label='Best Val Jaccard', color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, top10['final_val_jacard'], width,
                   label='Final Val Jaccard', color='#e74c3c', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Configuration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Jaccard Coefficient (IoU)', fontsize=13, fontweight='bold')
    ax.set_title('Top 10 Hyperparameter Configurations Performance',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(top10['config'], fontsize=9)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.2456, color='blue', linestyle='--', linewidth=2,
               label='Previous best (256×256): 0.2456', alpha=0.7)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_top10_configurations.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created: fig1_top10_configurations.png")

def plot_architecture_comparison(df):
    """Compare architectures"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot
    ax = axes[0]
    arch_order = ['unet', 'resunet', 'attention_resunet']
    bp = ax.boxplot([df[df['architecture'] == arch]['best_val_jacard'].values
                     for arch in arch_order],
                    labels=['U-Net', 'ResU-Net', 'Attention ResU-Net'],
                    patch_artist=True, showmeans=True)

    colors = ['#3498db', '#e74c3c', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Best Validation Jaccard', fontsize=12, fontweight='bold')
    ax.set_title('Architecture Performance Distribution', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Statistics table
    ax = axes[1]
    arch_stats = df.groupby('architecture')['best_val_jacard'].agg(['mean', 'std', 'max', 'count'])

    x_pos = np.arange(len(arch_stats))
    bars = ax.bar(x_pos, arch_stats['mean'], yerr=arch_stats['std'],
                  capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax.scatter(x_pos, arch_stats['max'], color='red', s=150, zorder=5,
               marker='*', label='Best', edgecolors='black', linewidths=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['U-Net\n(31M)', 'ResU-Net\n(33M)', 'Att-ResU-Net\n(34M)'])
    ax.set_ylabel('Jaccard Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Architecture Comparison (Mean ± Std)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    for i, (bar, mean, max_val) in enumerate(zip(bars, arch_stats['mean'], arch_stats['max'])):
        ax.text(bar.get_x() + bar.get_width()/2., mean,
               f'{mean:.3f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created: fig2_architecture_comparison.png")

def plot_batch_size_impact(df):
    """Analyze batch size impact"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Batch size performance
    ax = axes[0]
    bs_stats = df.groupby('batch_size')['best_val_jacard'].agg(['mean', 'std', 'max'])
    x_pos = np.arange(len(bs_stats))

    ax.bar(x_pos, bs_stats['mean'], yerr=bs_stats['std'], capsize=5,
           color='#16a085', alpha=0.7, edgecolor='black')
    ax.scatter(x_pos, bs_stats['max'], color='red', s=150, zorder=5,
               marker='*', label='Best')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'BS={int(bs)}' for bs in bs_stats.index])
    ax.set_ylabel('Jaccard Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Batch Size Impact', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Batch size vs overfitting
    ax = axes[1]
    df['overfitting'] = df['best_val_jacard'] - df['final_val_jacard']

    for bs in df['batch_size'].unique():
        data = df[df['batch_size'] == bs]
        ax.scatter(data['best_val_jacard'], data['overfitting'],
                  label=f'BS={int(bs)}', s=100, alpha=0.6, edgecolors='black', linewidths=1)

    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Best Validation Jaccard', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overfitting (Best - Final)', fontsize=12, fontweight='bold')
    ax.set_title('Batch Size vs Overfitting', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_batch_size_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created: fig3_batch_size_analysis.png")

def plot_loss_function_comparison(df):
    """Compare loss functions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    loss_order = ['focal', 'combined', 'focal_tversky', 'combined_tversky']

    # Violin plot
    ax = axes[0, 0]
    sns.violinplot(data=df, x='loss_function', y='best_val_jacard',
                   order=loss_order, ax=ax, palette='Set2', hue='loss_function', legend=False)
    sns.swarmplot(data=df, x='loss_function', y='best_val_jacard',
                  order=loss_order, ax=ax, color='black', alpha=0.5, size=5)
    ax.set_xlabel('Loss Function', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Validation Jaccard', fontsize=12, fontweight='bold')
    ax.set_title('Loss Function Performance Distribution', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(['Focal', 'Combined\n(D+F)', 'Focal\nTversky', 'Combined\nTversky'], fontsize=10)

    # Statistics
    ax = axes[0, 1]
    loss_stats = df.groupby('loss_function')['best_val_jacard'].agg(['mean', 'std', 'max'])
    loss_stats = loss_stats.reindex(loss_order)

    x_pos = np.arange(len(loss_stats))
    bars = ax.bar(x_pos, loss_stats['mean'], yerr=loss_stats['std'], capsize=5,
           color=['#8dd3c7', '#fb8072', '#bebada', '#fdb462'], alpha=0.7, edgecolor='black')
    ax.scatter(x_pos, loss_stats['max'], color='red', s=150, zorder=5, marker='*', label='Best')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Focal', 'Combined', 'F-Tversky', 'C-Tversky'])
    ax.set_ylabel('Jaccard Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Loss Function Statistics', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Overfitting by loss
    ax = axes[1, 0]
    for loss in loss_order:
        data = df[df['loss_function'] == loss]
        ax.scatter(data['best_val_jacard'], data['overfitting'],
                  label=loss, s=100, alpha=0.6, edgecolors='black', linewidths=1)

    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Best Validation Jaccard', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overfitting (Best - Final)', fontsize=12, fontweight='bold')
    ax.set_title('Loss Function vs Overfitting', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Count by architecture and loss
    ax = axes[1, 1]
    pivot = df.pivot_table(index='architecture', columns='loss_function',
                          values='best_val_jacard', aggfunc='mean')
    pivot = pivot.reindex(columns=loss_order)
    pivot = pivot.reindex(['unet', 'resunet', 'attention_resunet'])

    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.15, vmax=0.32)
    ax.set_xticks(np.arange(len(loss_order)))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(['Focal', 'Combined', 'F-Tversky', 'C-Tversky'])
    ax.set_yticklabels(['U-Net', 'ResU-Net', 'Att-ResU-Net'])
    ax.set_title('Mean Jaccard Heatmap', fontsize=13, fontweight='bold')

    for i in range(3):
        for j in range(len(loss_order)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       color='white' if val > 0.23 else 'black', fontweight='bold')

    plt.colorbar(im, ax=ax, label='Mean Jaccard')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_loss_function_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created: fig4_loss_function_analysis.png")

def plot_learning_curves(df):
    """Plot learning curves for best, median, and worst configurations"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Best
    best = df.iloc[0]
    best_history = load_training_history(
        best['architecture'], int(best['batch_size']),
        best['dropout'], best['loss_function']
    )

    if best_history is not None:
        ax = axes[0, 0]
        ax.plot(best_history['loss'], label='Train Loss', linewidth=2, color='#3498db')
        ax.plot(best_history['val_loss'], label='Val Loss', linewidth=2, color='#e74c3c')
        ax.axvline(x=best['best_epoch']-1, color='green', linestyle='--', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title(f"Best Config: Loss\n{best['architecture']}, BS={int(best['batch_size'])}, {best['loss_function']}",
                     fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.plot(best_history['jacard_coef'], label='Train Jaccard', linewidth=2, color='#2ecc71')
        ax.plot(best_history['val_jacard_coef'], label='Val Jaccard', linewidth=2, color='#f39c12')
        ax.axvline(x=best['best_epoch']-1, color='green', linestyle='--', linewidth=2)
        ax.axhline(y=best['best_val_jacard'], color='red', linestyle=':', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Jaccard Coefficient', fontsize=11, fontweight='bold')
        ax.set_title(f'Best: Peak Jaccard = {best["best_val_jacard"]:.3f}',
                     fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    # Median
    median_idx = len(df) // 2
    median = df.iloc[median_idx]
    median_history = load_training_history(
        median['architecture'], int(median['batch_size']),
        median['dropout'], median['loss_function']
    )

    if median_history is not None:
        ax = axes[0, 1]
        ax.plot(median_history['loss'], label='Train Loss', linewidth=2, color='#3498db')
        ax.plot(median_history['val_loss'], label='Val Loss', linewidth=2, color='#e74c3c')
        ax.axvline(x=median['best_epoch']-1, color='green', linestyle='--', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title(f"Median Config: Loss\n{median['architecture']}, BS={int(median['batch_size'])}, {median['loss_function']}",
                     fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.plot(median_history['jacard_coef'], label='Train Jaccard', linewidth=2, color='#2ecc71')
        ax.plot(median_history['val_jacard_coef'], label='Val Jaccard', linewidth=2, color='#f39c12')
        ax.axvline(x=median['best_epoch']-1, color='green', linestyle='--', linewidth=2)
        ax.axhline(y=median['best_val_jacard'], color='red', linestyle=':', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Jaccard Coefficient', fontsize=11, fontweight='bold')
        ax.set_title(f'Median: Peak Jaccard = {median["best_val_jacard"]:.3f}',
                     fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    # Worst
    worst = df.iloc[-1]
    worst_history = load_training_history(
        worst['architecture'], int(worst['batch_size']),
        worst['dropout'], worst['loss_function']
    )

    if worst_history is not None:
        ax = axes[0, 2]
        ax.plot(worst_history['loss'], label='Train Loss', linewidth=2, color='#3498db')
        ax.plot(worst_history['val_loss'], label='Val Loss', linewidth=2, color='#e74c3c')
        ax.axvline(x=worst['best_epoch']-1, color='green', linestyle='--', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title(f"Worst Config: Loss\n{worst['architecture']}, BS={int(worst['batch_size'])}, {worst['loss_function']}",
                     fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1, 2]
        ax.plot(worst_history['jacard_coef'], label='Train Jaccard', linewidth=2, color='#2ecc71')
        ax.plot(worst_history['val_jacard_coef'], label='Val Jaccard', linewidth=2, color='#f39c12')
        ax.axvline(x=worst['best_epoch']-1, color='green', linestyle='--', linewidth=2)
        ax.axhline(y=worst['best_val_jacard'], color='red', linestyle=':', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Jaccard Coefficient', fontsize=11, fontweight='bold')
        ax.set_title(f'Worst: Peak Jaccard = {worst["best_val_jacard"]:.3f}',
                     fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created: fig5_learning_curves.png")

def main():
    """Generate all visualizations"""
    print("="*80)
    print("Comprehensive Hyperparameter Search Analysis")
    print("="*80)
    print(f"\nResults directory: {RESULTS_DIR}")
    print()

    # Load results
    df = load_results()
    df['overfitting'] = df['best_val_jacard'] - df['final_val_jacard']
    print(f"✓ Loaded {len(df)} configurations\n")

    # Generate plots
    print("Generating visualizations...")
    print("-"*80)

    plot_top_configurations(df)
    plot_architecture_comparison(df)
    plot_batch_size_impact(df)
    plot_loss_function_comparison(df)
    plot_learning_curves(df)

    print("-"*80)
    print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - fig1_top10_configurations.png")
    print("  - fig2_architecture_comparison.png")
    print("  - fig3_batch_size_analysis.png")
    print("  - fig4_loss_function_analysis.png")
    print("  - fig5_learning_curves.png")
    print()

if __name__ == '__main__':
    main()
