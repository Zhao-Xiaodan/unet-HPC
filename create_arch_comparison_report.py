#!/usr/bin/env python3
"""
Create comprehensive architecture comparison report with visualizations
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11

results_dir = Path("validation_arch_comparison_20251013_093844")

# Load summary
with open(results_dir / "architecture_comparison_summary.json") as f:
    summary = json.load(f)

comparison = summary['comparison']
architectures = summary['architectures_tested']

arch_labels = {
    'unet': 'U-Net',
    'resunet': 'ResUNet',
    'attention_resunet': 'Attention ResUNet'
}

colors = {
    'unet': '#1f77b4',
    'resunet': '#ff7f0e',
    'attention_resunet': '#2ca02c'
}

# Load training histories
histories = {}
for arch in architectures:
    arch_dir = results_dir / arch
    histories[arch] = []
    for fold_dir in sorted(arch_dir.glob('fold_*')):
        history_file = fold_dir / 'history.csv'
        if history_file.exists():
            histories[arch].append(pd.read_csv(history_file))

print("Creating visualizations...")

# ============================================================================
# Figure 1: Performance Comparison
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Architecture Performance Comparison', fontsize=18, fontweight='bold', y=0.995)

# 1. Fold-by-Fold Performance
ax = axes[0, 0]
x = np.arange(5)
width = 0.25

for i, arch in enumerate(architectures):
    values = comparison[arch]['best_val_jacard']['values']
    offset = (i - 1) * width
    ax.bar(x + offset, values, width, label=arch_labels[arch],
           color=colors[arch], alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Fold', fontweight='bold', fontsize=12)
ax.set_ylabel('Best Val Jaccard', fontweight='bold', fontsize=12)
ax.set_title('(A) Best Validation Jaccard by Fold', fontweight='bold', fontsize=14, pad=10)
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i+1}' for i in range(len(x))])
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.8)

# 2. Box plot of performance distribution
ax = axes[0, 1]

data_for_box = []
labels_for_box = []
for arch in architectures:
    data_for_box.append(comparison[arch]['best_val_jacard']['values'])
    labels_for_box.append(arch_labels[arch])

bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                widths=0.6, showmeans=True, meanline=False,
                meanprops=dict(marker='D', markerfacecolor='red', markersize=10,
                              markeredgecolor='darkred', markeredgewidth=1.5))

for patch, arch in zip(bp['boxes'], architectures):
    patch.set_facecolor(colors[arch])
    patch.set_alpha(0.7)
    patch.set_linewidth(2)

ax.set_ylabel('Best Val Jaccard', fontweight='bold', fontsize=12)
ax.set_title('(B) Performance Distribution Across Folds', fontweight='bold', fontsize=14, pad=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.8)

# Add mean line
for i, arch in enumerate(architectures):
    mean_val = comparison[arch]['best_val_jacard']['mean']
    ax.hlines(mean_val, i+0.65, i+1.35, colors='red', linestyles='--', linewidth=2)

# 3. Training Time Comparison
ax = axes[1, 0]

arch_names = [arch_labels[a] for a in architectures]
epoch_times = [comparison[a]['avg_epoch_time_sec']['mean'] for a in architectures]
epoch_stds = [comparison[a]['avg_epoch_time_sec']['std'] for a in architectures]

x_pos = np.arange(len(arch_names))
bars = ax.bar(x_pos, epoch_times, yerr=epoch_stds, capsize=8,
              color=[colors[a] for a in architectures], alpha=0.8,
              edgecolor='black', linewidth=1.5)

ax.set_xlabel('Architecture', fontweight='bold', fontsize=12)
ax.set_ylabel('Average Epoch Time (seconds)', fontweight='bold', fontsize=12)
ax.set_title('(C) Training Speed Comparison', fontweight='bold', fontsize=14, pad=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(arch_names)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, time in zip(bars, epoch_times):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
           f'{time:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. Overfitting Gap Comparison
ax = axes[1, 1]

gaps = [comparison[a]['overfitting_gap']['mean'] for a in architectures]
gap_stds = [comparison[a]['overfitting_gap']['std'] for a in architectures]

bars = ax.bar(x_pos, gaps, yerr=gap_stds, capsize=8,
              color=[colors[a] for a in architectures], alpha=0.8,
              edgecolor='black', linewidth=1.5)

ax.set_xlabel('Architecture', fontweight='bold', fontsize=12)
ax.set_ylabel('Overfitting Gap (Train/Val Jaccard)', fontweight='bold', fontsize=12)
ax.set_title('(D) Generalization: Overfitting Gap', fontweight='bold', fontsize=14, pad=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(arch_names)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=2, label='No gap (perfect)')

# Add value labels
for bar, gap in zip(bars, gaps):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
           f'{gap:.2f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(results_dir / 'arch_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: arch_performance_comparison.png")
plt.close()

# ============================================================================
# Figure 2: Training Curves
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Training Curves Comparison Across Architectures', fontsize=18, fontweight='bold')

for idx, arch in enumerate(architectures):
    ax = axes[idx]

    for fold_idx, df in enumerate(histories[arch]):
        epochs = range(len(df))

        # Plot training (light) and validation (dark)
        ax.plot(epochs, df['jacard_coef'], color=colors[arch],
               alpha=0.25, linewidth=1.5, linestyle='--')
        ax.plot(epochs, df['val_jacard_coef'], color=colors[arch],
               alpha=0.7, linewidth=2.5, label=f'Fold {fold_idx+1}')

        # Mark best epoch
        best_epoch = df['val_jacard_coef'].idxmax()
        best_val = df['val_jacard_coef'].max()
        ax.plot(best_epoch, best_val, '*', color='gold',
               markersize=16, markeredgecolor='black', markeredgewidth=1, zorder=10)

    ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax.set_ylabel('Jaccard Coefficient', fontweight='bold', fontsize=12)
    ax.set_title(f'{arch_labels[arch]}', fontweight='bold', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0, 1.0)

    # Add text explanation
    ax.text(0.02, 0.98, 'Dashed: Training\nSolid: Validation\n⭐: Best epoch',
           transform=ax.transAxes, fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))

plt.tight_layout()
plt.savefig(results_dir / 'arch_training_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: arch_training_curves.png")
plt.close()

# ============================================================================
# Figure 3: Convergence Analysis
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Convergence and Learning Dynamics Analysis', fontsize=18, fontweight='bold', y=0.995)

# 1. Average validation curves
ax = axes[0, 0]

for arch in architectures:
    max_epochs = max(len(df) for df in histories[arch])

    padded = []
    for df in histories[arch]:
        vals = df['val_jacard_coef'].values
        if len(vals) < max_epochs:
            vals = np.pad(vals, (0, max_epochs - len(vals)), mode='edge')
        padded.append(vals)

    mean_curve = np.mean(padded, axis=0)
    std_curve = np.std(padded, axis=0)
    epochs = range(max_epochs)

    ax.plot(epochs, mean_curve, label=arch_labels[arch],
           color=colors[arch], linewidth=3)
    ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve,
                   color=colors[arch], alpha=0.2)

ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
ax.set_ylabel('Val Jaccard (Mean ± Std)', fontweight='bold', fontsize=12)
ax.set_title('(A) Average Convergence Curves', fontweight='bold', fontsize=14, pad=10)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim(0, 0.8)

# 2. Best Epoch Distribution
ax = axes[0, 1]

best_epochs = {arch: comparison[arch]['best_epoch']['values'] for arch in architectures}

positions = []
data_for_violin = []
for arch in architectures:
    data_for_violin.append(best_epochs[arch])

parts = ax.violinplot(data_for_violin, positions=range(len(architectures)),
                     showmeans=True, showextrema=True, widths=0.7)

for i, arch in enumerate(architectures):
    parts['bodies'][i].set_facecolor(colors[arch])
    parts['bodies'][i].set_alpha(0.7)

ax.set_ylabel('Best Epoch', fontweight='bold', fontsize=12)
ax.set_title('(B) Best Epoch Distribution', fontweight='bold', fontsize=14, pad=10)
ax.set_xticks(range(len(architectures)))
ax.set_xticklabels([arch_labels[a] for a in architectures])
ax.grid(axis='y', alpha=0.3)

# 3. Overfitting progression
ax = axes[1, 0]

for arch in architectures:
    gaps = []
    for df in histories[arch]:
        train_vals = df['jacard_coef'].values
        val_vals = df['val_jacard_coef'].values
        gap = train_vals / np.maximum(val_vals, 1e-6)
        gaps.append(gap)

    max_epochs = max(len(g) for g in gaps)
    padded_gaps = []
    for gap in gaps:
        if len(gap) < max_epochs:
            gap = np.pad(gap, (0, max_epochs - len(gap)), mode='edge')
        padded_gaps.append(gap)

    mean_gap = np.mean(padded_gaps, axis=0)
    std_gap = np.std(padded_gaps, axis=0)
    epochs = range(max_epochs)

    ax.plot(epochs, mean_gap, label=arch_labels[arch],
           color=colors[arch], linewidth=3)
    ax.fill_between(epochs, mean_gap - std_gap, mean_gap + std_gap,
                   color=colors[arch], alpha=0.2)

ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
ax.set_ylabel('Overfitting Gap (Train/Val Jaccard)', fontweight='bold', fontsize=12)
ax.set_title('(C) Overfitting Progression Over Training', fontweight='bold', fontsize=14, pad=10)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, linewidth=2)
ax.set_ylim(0, 8)

# 4. Final Performance Summary Table
ax = axes[1, 1]
ax.axis('off')

table_data = []
table_data.append(['Metric', 'U-Net', 'ResUNet', 'Attention ResUNet'])
table_data.append(['─' * 20, '─' * 12, '─' * 12, '─' * 18])

# Best Val Jaccard
row = ['Best Val Jaccard']
for arch in architectures:
    mean = comparison[arch]['best_val_jacard']['mean']
    std = comparison[arch]['best_val_jacard']['std']
    row.append(f'{mean:.4f} ± {std:.4f}')
table_data.append(row)

# Improvement vs U-Net
row = ['vs U-Net (%)']
unet_mean = comparison['unet']['best_val_jacard']['mean']
row.append('baseline')
for arch in architectures[1:]:
    mean = comparison[arch]['best_val_jacard']['mean']
    improvement = ((mean - unet_mean) / unet_mean) * 100
    row.append(f'{improvement:+.1f}%')
table_data.append(row)

# Best Epoch
row = ['Best Epoch']
for arch in architectures:
    mean = comparison[arch]['best_epoch']['mean']
    std = comparison[arch]['best_epoch']['std']
    row.append(f'{mean:.1f} ± {std:.1f}')
table_data.append(row)

# Overfitting Gap
row = ['Overfitting Gap']
for arch in architectures:
    mean = comparison[arch]['overfitting_gap']['mean']
    std = comparison[arch]['overfitting_gap']['std']
    row.append(f'{mean:.2f}× ± {std:.2f}×')
table_data.append(row)

# Training Time
row = ['Epoch Time (s)']
for arch in architectures:
    mean = comparison[arch]['avg_epoch_time_sec']['mean']
    row.append(f'{mean:.1f}s')
table_data.append(row)

# Parameters
row = ['Parameters (M)']
for arch in architectures:
    params = comparison[arch]['total_parameters'] / 1e6
    row.append(f'{params:.1f}M')
table_data.append(row)

table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                colWidths=[0.3, 0.23, 0.23, 0.24])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#404040')
    cell.set_text_props(weight='bold', color='white')

# Style data rows
for i in range(2, len(table_data)):
    for j in range(4):
        cell = table[(i, j)]
        if j == 0:
            cell.set_facecolor('#f0f0f0')
            cell.set_text_props(weight='bold')
        else:
            arch = architectures[j-1]
            cell.set_facecolor(colors[arch])
            cell.set_alpha(0.3)

ax.set_title('(D) Performance Summary Table', fontweight='bold', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig(results_dir / 'arch_convergence_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: arch_convergence_analysis.png")
plt.close()

# ============================================================================
# Statistical Tests
# ============================================================================

print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE TESTING")
print("="*80)

stat_results = []

for i, arch1 in enumerate(architectures):
    for arch2 in architectures[i+1:]:
        data1 = comparison[arch1]['best_val_jacard']['values']
        data2 = comparison[arch2]['best_val_jacard']['values']

        t_stat, p_value = stats.ttest_rel(data1, data2)

        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        diff = mean2 - mean1
        diff_pct = (diff / mean1) * 100

        print(f"\n{arch_labels[arch1]} vs {arch_labels[arch2]}:")
        print(f"  Mean difference: {diff:+.4f} ({diff_pct:+.1f}%)")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            winner = arch_labels[arch2] if diff > 0 else arch_labels[arch1]
            print(f"  ✅ SIGNIFICANT: {winner} performs better (p < 0.05)")
        else:
            print(f"  ❌ NOT SIGNIFICANT (p ≥ 0.05)")

        stat_results.append({
            'comparison': f'{arch_labels[arch1]} vs {arch_labels[arch2]}',
            'mean_diff': float(diff),
            'diff_pct': float(diff_pct),
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        })

print("\n" + "="*80)

print("\n✅ All visualizations created successfully!")
print(f"\nGenerated files in {results_dir}:")
print("  - arch_performance_comparison.png")
print("  - arch_training_curves.png")
print("  - arch_convergence_analysis.png")
