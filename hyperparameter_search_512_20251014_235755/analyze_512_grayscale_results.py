#!/usr/bin/env python3
"""
Analyze 512×512 Grayscale Hyperparameter Search Results
========================================================

Analyzes results from hyperparameter_search_512_20251014_235755/
Generates comprehensive report with figures.

Date: October 15, 2025
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

# Configuration
RESULTS_DIR = Path('hyperparameter_search_512_20251014_235755')
OUTPUT_DIR = RESULTS_DIR
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

print("="*80)
print("512×512 GRAYSCALE HYPERPARAMETER SEARCH ANALYSIS")
print("="*80)
print(f"Results directory: {RESULTS_DIR}")
print(f"Figures will be saved to: {FIGURES_DIR}")
print()

# Load data
print("Loading data...")
df = pd.read_csv(RESULTS_DIR / 'all_results.csv')

# Parse config string
def parse_config(config_str):
    """Parse config dictionary string"""
    import ast
    return ast.literal_eval(config_str)

df['config_dict'] = df['config'].apply(parse_config)
df['learning_rate'] = df['config_dict'].apply(lambda x: x['learning_rate'])
df['dropout'] = df['config_dict'].apply(lambda x: x['dropout'])
df['batch_size'] = df['config_dict'].apply(lambda x: x['batch_size'])

print(f"✓ Loaded {len(df)} training runs")
print(f"  Configurations: {df['config_name'].nunique()}")
print(f"  Architectures: {df['architecture'].nunique()}")
print(f"  Folds: {df['fold'].nunique()}")
print()

# Summary statistics
print("="*80)
print("OVERALL STATISTICS")
print("="*80)
print(f"Total runs: {len(df)}")
print(f"Successful runs: {df['success'].sum()} ({df['success'].mean()*100:.1f}%)")
print(f"Mean Jaccard: {df['best_val_jaccard'].mean():.4f} ± {df['best_val_jaccard'].std():.4f}")
print(f"Best Jaccard: {df['best_val_jaccard'].max():.4f}")
print(f"Worst Jaccard: {df['best_val_jaccard'].min():.4f}")
print()

# ============================================================================
# FIGURE 1: Overall Performance Distribution
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Figure 1: Overall Performance Distribution - 512×512 Grayscale Images',
             fontsize=14, fontweight='bold')

# 1A: Histogram of Jaccard scores
ax = axes[0, 0]
ax.hist(df['best_val_jaccard'], bins=20, edgecolor='black', alpha=0.7)
ax.axvline(df['best_val_jaccard'].mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {df["best_val_jaccard"].mean():.4f}')
ax.axvline(df['best_val_jaccard'].median(), color='orange', linestyle='--',
           linewidth=2, label=f'Median: {df["best_val_jaccard"].median():.4f}')
ax.set_xlabel('Best Validation Jaccard', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(A) Distribution of Best Jaccard Scores', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 1B: Box plot by architecture
ax = axes[0, 1]
arch_data = [df[df['architecture']==arch]['best_val_jaccard'].values
             for arch in ['unet', 'resunet', 'attention_resunet']]
bp = ax.boxplot(arch_data, labels=['U-Net', 'ResUNet', 'Attention\nResUNet'],
                patch_artist=True, showmeans=True)
for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('Best Validation Jaccard', fontsize=11)
ax.set_title('(B) Performance by Architecture', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 1C: Training vs Validation Jaccard (overfitting check)
ax = axes[1, 0]
for arch, color in zip(['unet', 'resunet', 'attention_resunet'],
                       ['#1f77b4', '#ff7f0e', '#2ca02c']):
    mask = df['architecture'] == arch
    ax.scatter(df[mask]['final_train_jaccard'], df[mask]['best_val_jaccard'],
               label=arch.replace('_', ' ').title(), alpha=0.6, s=50, color=color)
ax.plot([0, 0.7], [0, 0.7], 'k--', alpha=0.5, label='Perfect fit')
ax.set_xlabel('Final Train Jaccard', fontsize=11)
ax.set_ylabel('Best Validation Jaccard', fontsize=11)
ax.set_title('(C) Train vs Validation Performance', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 1D: Overfitting gap distribution
ax = axes[1, 1]
for arch, color in zip(['unet', 'resunet', 'attention_resunet'],
                       ['#1f77b4', '#ff7f0e', '#2ca02c']):
    mask = df['architecture'] == arch
    ax.hist(df[mask]['overfitting_gap'], bins=15, alpha=0.5,
            label=arch.replace('_', ' ').title(), edgecolor='black', color=color)
ax.axvline(10, color='red', linestyle='--', linewidth=2,
           label='10% threshold', alpha=0.7)
ax.set_xlabel('Overfitting Gap (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(D) Overfitting Gap Distribution', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure1_overall_performance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved Figure 1: {FIGURES_DIR / 'figure1_overall_performance.png'}")

# ============================================================================
# FIGURE 2: Architecture Comparison
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Figure 2: Architecture Comparison', fontsize=14, fontweight='bold')

# 2A: Mean performance by architecture
ax = axes[0, 0]
arch_summary = df.groupby('architecture')['best_val_jaccard'].agg(['mean', 'std', 'count'])
arch_names = ['U-Net', 'ResUNet', 'Attention\nResUNet']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
x_pos = np.arange(len(arch_names))
bars = ax.bar(x_pos, arch_summary['mean'], yerr=arch_summary['std'],
              color=colors, alpha=0.7, capsize=10, edgecolor='black', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(arch_names)
ax.set_ylabel('Mean Jaccard ± Std', fontsize=11)
ax.set_title('(A) Mean Performance by Architecture', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
# Add value labels
for i, (bar, val) in enumerate(zip(bars, arch_summary['mean'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2B: Best configuration per architecture
ax = axes[0, 1]
best_per_arch = df.groupby('architecture')['best_val_jaccard'].max()
bars = ax.bar(x_pos, best_per_arch, color=colors, alpha=0.7,
              edgecolor='black', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(arch_names)
ax.set_ylabel('Best Jaccard', fontsize=11)
ax.set_title('(B) Best Score per Architecture', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, best_per_arch)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2C: Epochs to best performance
ax = axes[1, 0]
arch_data = [df[df['architecture']==arch]['best_epoch'].values
             for arch in ['unet', 'resunet', 'attention_resunet']]
bp = ax.boxplot(arch_data, labels=arch_names, patch_artist=True, showmeans=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('Best Epoch', fontsize=11)
ax.set_title('(C) Convergence Speed', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 2D: Consistency (std across folds)
ax = axes[1, 1]
consistency = df.groupby(['architecture', 'config_name'])['best_val_jaccard'].std().reset_index()
consistency_mean = consistency.groupby('architecture')['best_val_jaccard'].mean()
bars = ax.bar(x_pos, consistency_mean, color=colors, alpha=0.7,
              edgecolor='black', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(arch_names)
ax.set_ylabel('Mean Std Across Configs', fontsize=11)
ax.set_title('(D) Cross-Fold Consistency (lower = better)', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, consistency_mean)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure2_architecture_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved Figure 2: {FIGURES_DIR / 'figure2_architecture_comparison.png'}")

# ============================================================================
# FIGURE 3: Hyperparameter Effects
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Figure 3: Hyperparameter Effects on Performance',
             fontsize=14, fontweight='bold')

# 3A: Learning rate effect
ax = axes[0, 0]
lr_summary = df.groupby('learning_rate')['best_val_jaccard'].agg(['mean', 'std', 'count'])
lr_labels = ['1e-04', '5e-05']
x_pos = np.arange(len(lr_labels))
bars = ax.bar(x_pos, lr_summary['mean'], yerr=lr_summary['std'],
              color=['#e74c3c', '#3498db'], alpha=0.7, capsize=10,
              edgecolor='black', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(lr_labels)
ax.set_xlabel('Learning Rate', fontsize=11)
ax.set_ylabel('Mean Jaccard ± Std', fontsize=11)
ax.set_title('(A) Learning Rate Effect', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, lr_summary['mean'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}\n(n={int(lr_summary["count"].iloc[i])//3}×3)',
            ha='center', va='bottom', fontsize=9)

# 3B: Dropout effect
ax = axes[0, 1]
dropout_summary = df.groupby('dropout')['best_val_jaccard'].agg(['mean', 'std', 'count'])
dropout_labels = ['0.2', '0.3']
x_pos = np.arange(len(dropout_labels))
bars = ax.bar(x_pos, dropout_summary['mean'], yerr=dropout_summary['std'],
              color=['#9b59b6', '#1abc9c'], alpha=0.7, capsize=10,
              edgecolor='black', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(dropout_labels)
ax.set_xlabel('Dropout Rate', fontsize=11)
ax.set_ylabel('Mean Jaccard ± Std', fontsize=11)
ax.set_title('(B) Dropout Effect', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, dropout_summary['mean'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}\n(n={int(dropout_summary["count"].iloc[i])//3}×3)',
            ha='center', va='bottom', fontsize=9)

# 3C: Heatmap - Architecture × Learning Rate
ax = axes[1, 0]
pivot_lr = df.pivot_table(values='best_val_jaccard',
                           index='architecture',
                           columns='learning_rate',
                           aggfunc='mean')
sns.heatmap(pivot_lr, annot=True, fmt='.4f', cmap='YlOrRd',
            cbar_kws={'label': 'Mean Jaccard'}, ax=ax,
            yticklabels=['U-Net', 'ResUNet', 'Attention\nResUNet'],
            xticklabels=['1e-04', '5e-05'])
ax.set_xlabel('Learning Rate', fontsize=11)
ax.set_ylabel('Architecture', fontsize=11)
ax.set_title('(C) Architecture × Learning Rate', fontsize=12)

# 3D: Heatmap - Architecture × Dropout
ax = axes[1, 1]
pivot_drop = df.pivot_table(values='best_val_jaccard',
                             index='architecture',
                             columns='dropout',
                             aggfunc='mean')
sns.heatmap(pivot_drop, annot=True, fmt='.4f', cmap='YlGnBu',
            cbar_kws={'label': 'Mean Jaccard'}, ax=ax,
            yticklabels=['U-Net', 'ResUNet', 'Attention\nResUNet'])
ax.set_xlabel('Dropout', fontsize=11)
ax.set_ylabel('Architecture', fontsize=11)
ax.set_title('(D) Architecture × Dropout', fontsize=12)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure3_hyperparameter_effects.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved Figure 3: {FIGURES_DIR / 'figure3_hyperparameter_effects.png'}")

# ============================================================================
# FIGURE 4: Top Configurations and Training Curves
# ============================================================================

# Get top 5 configurations
config_summary = df.groupby('config_name')['best_val_jaccard'].agg(['mean', 'std']).sort_values('mean', ascending=False)
top5_configs = config_summary.head(5).index.tolist()

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
fig.suptitle('Figure 4: Top 5 Configurations and Training Curves',
             fontsize=14, fontweight='bold')

# 4A: Top 5 configurations bar chart
ax = fig.add_subplot(gs[0, :])
top5_data = config_summary.head(5)
x_pos = np.arange(5)
bars = ax.barh(x_pos, top5_data['mean'], xerr=top5_data['std'],
               color=plt.cm.viridis(np.linspace(0.2, 0.8, 5)),
               alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)
ax.set_yticks(x_pos)
ax.set_yticklabels([c.replace('_', ' ') for c in top5_configs], fontsize=9)
ax.set_xlabel('Mean Jaccard ± Std (3-fold CV)', fontsize=11)
ax.set_title('(A) Top 5 Configurations', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')
# Add value labels
for bar, val, std in zip(bars, top5_data['mean'], top5_data['std']):
    ax.text(val + std + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

# 4B-F: Training curves for top 5 configs (show fold 1 only)
for idx, config in enumerate(top5_configs):
    row = 1 + idx // 3
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])

    # Find fold 1 history file
    fold1_files = [f for f in RESULTS_DIR.glob(f'{config.replace("_lr", "_fold1_lr")}*_history.csv')]
    if fold1_files:
        history = pd.read_csv(fold1_files[0])
        ax.plot(history['epoch'], history['jacard_coef'], 'b-', linewidth=2, label='Train')
        ax.plot(history['epoch'], history['val_jacard_coef'], 'r-', linewidth=2, label='Val')

        # Mark best epoch
        best_idx = history['val_jacard_coef'].idxmax()
        ax.axvline(history.loc[best_idx, 'epoch'], color='green',
                   linestyle='--', alpha=0.7, label=f'Best: {best_idx}')

        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('Jaccard', fontsize=9)
        short_name = config.replace('_lr', ' lr').replace('_drop', ' d').replace('_bs', ' bs')
        ax.set_title(f'({chr(66+idx)}) {short_name}', fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

plt.savefig(FIGURES_DIR / 'figure4_top_configs_curves.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved Figure 4: {FIGURES_DIR / 'figure4_top_configs_curves.png'}")

# ============================================================================
# FIGURE 5: Comparison with Previous RGB Results
# ============================================================================

# Previous 512×512 RGB results (from hyperparameter_search_512_20251014_142259)
previous_best = 0.1562  # attention_resunet_lr5e-05_drop0.2_bs4
previous_mean = 0.1416  # Overall mean

# Current 512×512 Grayscale results
current_best = df['best_val_jaccard'].max()
current_mean = df['best_val_jaccard'].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Figure 5: Grayscale vs RGB Comparison (512×512)',
             fontsize=14, fontweight='bold')

# 5A: Best performance comparison
ax = axes[0]
comparison_data = {
    'RGB\n(FP16+nan)': previous_best,
    'Grayscale\n(FP32)': current_best
}
colors = ['#e74c3c', '#27ae60']
bars = ax.bar(comparison_data.keys(), comparison_data.values(),
              color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Best Jaccard Score', fontsize=11)
ax.set_title('(A) Best Performance Comparison', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, comparison_data.values()):
    improvement = ((current_best - previous_best) / previous_best * 100)
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
if improvement > 0:
    ax.text(0.5, max(comparison_data.values()) * 0.9,
            f'+{improvement:.1f}% improvement', ha='center',
            fontsize=12, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# 5B: Mean performance comparison
ax = axes[1]
comparison_data = {
    'RGB\n(FP16+nan)': previous_mean,
    'Grayscale\n(FP32)': current_mean
}
bars = ax.bar(comparison_data.keys(), comparison_data.values(),
              color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Mean Jaccard Score', fontsize=11)
ax.set_title('(B) Mean Performance Comparison', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, comparison_data.values()):
    improvement = ((current_mean - previous_mean) / previous_mean * 100)
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
if improvement > 0:
    ax.text(0.5, max(comparison_data.values()) * 0.9,
            f'+{improvement:.1f}% improvement', ha='center',
            fontsize=12, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure5_rgb_vs_grayscale.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved Figure 5: {FIGURES_DIR / 'figure5_rgb_vs_grayscale.png'}")

# ============================================================================
# Generate Summary Tables
# ============================================================================

print("\n" + "="*80)
print("DETAILED RESULTS")
print("="*80)

# Best configuration
best_idx = df['best_val_jaccard'].idxmax()
best_row = df.loc[best_idx]
print(f"\n🏆 BEST SINGLE RUN:")
print(f"   Config: {best_row['config_name']}")
print(f"   Fold: {best_row['fold']}")
print(f"   Jaccard: {best_row['best_val_jaccard']:.4f}")
print(f"   Architecture: {best_row['architecture']}")
print(f"   LR: {best_row['learning_rate']}, Dropout: {best_row['dropout']}")

# Best average configuration
print(f"\n🎯 BEST AVERAGE CONFIGURATION (3-fold CV):")
best_config = config_summary.index[0]
best_config_mean = config_summary.iloc[0]['mean']
best_config_std = config_summary.iloc[0]['std']
print(f"   Config: {best_config}")
print(f"   Mean Jaccard: {best_config_mean:.4f} ± {best_config_std:.4f}")

# Architecture ranking
print(f"\n📊 ARCHITECTURE RANKING:")
arch_ranking = df.groupby('architecture')['best_val_jaccard'].mean().sort_values(ascending=False)
for rank, (arch, score) in enumerate(arch_ranking.items(), 1):
    print(f"   {rank}. {arch.replace('_', ' ').title()}: {score:.4f}")

# Hyperparameter insights
print(f"\n🔍 HYPERPARAMETER INSIGHTS:")
print(f"   Best Learning Rate: {df.groupby('learning_rate')['best_val_jaccard'].mean().idxmax()}")
print(f"   Best Dropout: {df.groupby('dropout')['best_val_jaccard'].mean().idxmax()}")

print(f"\n✓ Analysis complete! All figures saved to: {FIGURES_DIR}")
print("="*80)
