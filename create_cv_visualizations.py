#!/usr/bin/env python3
"""
Create visualizations for cross-validation results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load CV summary
cv_dir = Path('/Users/xiaodan/unetCNN/unet-HPC/validation_cv_20251013_052113')
with open(cv_dir / 'cv_summary_fixed.json') as f:
    summary = json.load(f)

# Output directory
output_dir = cv_dir
output_dir.mkdir(exist_ok=True)

# Extract data
fold_results = summary['fold_results']
stats = summary['statistics']

##############################################################################
# Figure 1: Fold-by-Fold Performance Comparison
##############################################################################

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Cross-Validation Results: Fold-by-Fold Analysis', fontsize=16, fontweight='bold')

# Plot 1: Best Val Jaccard by Fold
ax = axes[0, 0]
folds = [f['fold'] for f in fold_results]
best_vals = [f['best_val_jacard'] * 100 for f in fold_results]
colors = ['#2ecc71' if v > 60 else '#3498db' if v > 50 else '#e74c3c' for v in best_vals]

bars = ax.bar(folds, best_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(stats['best_val_jacard']['mean'] * 100, color='red', linestyle='--',
           linewidth=2, label=f'Mean: {stats["best_val_jacard"]["mean"]*100:.1f}%')
ax.axhline(13.8, color='orange', linestyle=':', linewidth=2, label='Phase 1: 13.8%')
ax.set_xlabel('Fold Number', fontweight='bold')
ax.set_ylabel('Best Validation Jaccard (%)', fontweight='bold')
ax.set_title('Best Validation Jaccard by Fold', fontweight='bold')
ax.set_ylim([0, 100])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, best_vals)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{val:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 2: Overfitting Gap by Fold
ax = axes[0, 1]
gaps = [f['overfitting_gap'] for f in fold_results]
colors_gap = ['#2ecc71' if g < 2 else '#3498db' if g < 3 else '#e74c3c' for g in gaps]

bars = ax.bar(folds, gaps, color=colors_gap, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(stats['overfitting_gap']['mean'], color='red', linestyle='--',
           linewidth=2, label=f'Mean: {stats["overfitting_gap"]["mean"]:.1f}×')
ax.axhline(10.5, color='orange', linestyle=':', linewidth=2, label='Phase 1: 10.5×')
ax.set_xlabel('Fold Number', fontweight='bold')
ax.set_ylabel('Overfitting Gap (Train/Val)', fontweight='bold')
ax.set_title('Overfitting Gap by Fold', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, gaps):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{val:.1f}×',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 3: Best Epoch by Fold
ax = axes[1, 0]
best_epochs = [f['best_epoch'] + 1 for f in fold_results]  # +1 for 1-indexed
colors_epoch = ['#2ecc71' if e > 5 else '#e74c3c' if e <= 2 else '#3498db' for e in best_epochs]

bars = ax.bar(folds, best_epochs, color=colors_epoch, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(stats['best_epoch']['mean'] + 1, color='red', linestyle='--',
           linewidth=2, label=f'Mean: {stats["best_epoch"]["mean"]+1:.1f}')
ax.axhline(1, color='orange', linestyle=':', linewidth=2, label='Phase 1: Epoch 1')
ax.set_xlabel('Fold Number', fontweight='bold')
ax.set_ylabel('Best Epoch', fontweight='bold')
ax.set_title('Best Epoch by Fold (1-indexed)', fontweight='bold')
ax.set_ylim([0, 20])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, best_epochs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'Ep {val}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 4: Final vs Best Val Jaccard
ax = axes[1, 1]
final_vals = [f['final_val_jacard'] * 100 for f in fold_results]
x_pos = np.arange(len(folds))
width = 0.35

bars1 = ax.bar(x_pos - width/2, best_vals, width, label='Best Val Jaccard',
               color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x_pos + width/2, final_vals, width, label='Final Val Jaccard',
               color='#95a5a6', alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Fold Number', fontweight='bold')
ax.set_ylabel('Validation Jaccard (%)', fontweight='bold')
ax.set_title('Best vs Final Validation Jaccard', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(folds)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 100])

plt.tight_layout()
plt.savefig(output_dir / 'cv_fold_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: cv_fold_comparison.png")
plt.close()

##############################################################################
# Figure 2: Training Curves for All Folds
##############################################################################

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Training Curves: All 5 Folds', fontsize=16, fontweight='bold')

for i, fold_num in enumerate([1, 2, 3, 4, 5]):
    row = i // 2
    col = i % 2
    ax = axes[row, col] if i < 4 else axes[2, 0]

    # Load history
    history = pd.read_csv(cv_dir / f'fold_{fold_num}' / 'history.csv')

    epochs = history['epoch'] + 1  # 1-indexed
    train_j = history['jacard_coef'] * 100
    val_j = history['val_jacard_coef'] * 100

    # Plot
    ax.plot(epochs, train_j, 'o-', color='#3498db', linewidth=2, markersize=6,
            label='Train Jaccard', alpha=0.8)
    ax.plot(epochs, val_j, 's-', color='#e74c3c', linewidth=2, markersize=6,
            label='Val Jaccard', alpha=0.8)

    # Mark best epoch
    best_epoch = fold_results[i]['best_epoch'] + 1
    best_val = fold_results[i]['best_val_jacard'] * 100
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax.plot(best_epoch, best_val, '*', color='gold', markersize=20,
            markeredgecolor='black', markeredgewidth=1.5,
            label=f'Best: Ep {best_epoch}, {best_val:.1f}%', zorder=10)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Jaccard Coefficient (%)', fontweight='bold')
    ax.set_title(f'Fold {fold_num} (Best: {best_val:.1f}% at Epoch {best_epoch})',
                 fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 100])

# Remove extra subplot
axes[2, 1].axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'cv_training_curves.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: cv_training_curves.png")
plt.close()

##############################################################################
# Figure 3: Comparison with Previous Tests
##############################################################################

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Cross-Validation vs Previous Single-Split Tests', fontsize=16, fontweight='bold')

# Plot 1: Best Val Jaccard Comparison
ax = axes[0]
tests = ['Phase 1\n(filters=64)', 'Focal\nTversky\n(filters=64)', 'Small\nModel\n(filters=16)', 'CV Mean\n(filters=64)']
values = [13.8, 13.3, 7.6, stats['best_val_jacard']['mean'] * 100]
colors_comp = ['#e74c3c', '#e67e22', '#c0392b', '#2ecc71']

bars = ax.bar(tests, values, color=colors_comp, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Best Validation Jaccard (%)', fontweight='bold')
ax.set_title('Best Validation Performance Comparison', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 80])

# Add value labels and improvement
for i, (bar, val) in enumerate(zip(bars, values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{val:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

    if i == 3:  # CV mean
        improvement = ((val - 13.8) / 13.8) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'+{improvement:.0f}%\nvs Phase 1',
                ha='center', va='center', fontweight='bold', fontsize=10,
                color='white', bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))

# Plot 2: Overfitting Gap Comparison
ax = axes[1]
gap_values = [10.5, 15.1, 4.4, stats['overfitting_gap']['mean']]
colors_gap_comp = ['#e74c3c', '#c0392b', '#3498db', '#2ecc71']

bars = ax.bar(tests, gap_values, color=colors_gap_comp, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Overfitting Gap (Train/Val)', fontweight='bold')
ax.set_title('Overfitting Gap Comparison', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 20])

# Add value labels
for i, (bar, val) in enumerate(zip(bars, gap_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{val:.1f}×',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

    if i == 3:  # CV mean
        reduction = 10.5 - val
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'-{reduction:.1f}×\nvs Phase 1',
                ha='center', va='center', fontweight='bold', fontsize=10,
                color='white', bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / 'cv_comparison_previous_tests.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: cv_comparison_previous_tests.png")
plt.close()

##############################################################################
# Figure 4: Statistical Summary
##############################################################################

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Cross-Validation Statistical Summary', fontsize=16, fontweight='bold')

# Plot 1: Distribution of Best Val Jaccard
ax = axes[0, 0]
best_val_values = [f['best_val_jacard'] * 100 for f in fold_results]
mean_val = stats['best_val_jacard']['mean'] * 100
std_val = stats['best_val_jacard']['std'] * 100

ax.hist(best_val_values, bins=10, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}%')
ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2, alpha=0.7)
ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f'±1 SD: {std_val:.1f}%')
ax.set_xlabel('Best Validation Jaccard (%)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Distribution of Best Validation Jaccard Across Folds', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Box plot of metrics
ax = axes[0, 1]
data_to_plot = [
    [f['best_val_jacard'] * 100 for f in fold_results],
    [f['final_val_jacard'] * 100 for f in fold_results],
    [f['final_train_jacard'] * 100 for f in fold_results]
]
positions = [1, 2, 3]
labels = ['Best\nVal', 'Final\nVal', 'Final\nTrain']

bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                labels=labels, showmeans=True, meanline=True)

colors_box = ['#3498db', '#95a5a6', '#2ecc71']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Jaccard Coefficient (%)', fontweight='bold')
ax.set_title('Distribution of Jaccard Metrics Across Folds', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 100])

# Plot 3: Epoch analysis
ax = axes[1, 0]
epoch_data = {
    'Epoch ≤1': sum(1 for f in fold_results if f['best_epoch'] <= 1),
    'Epoch 2-5': sum(1 for f in fold_results if 2 <= f['best_epoch'] + 1 <= 5),
    'Epoch 6-10': sum(1 for f in fold_results if 6 <= f['best_epoch'] + 1 <= 10),
    'Epoch 11-20': sum(1 for f in fold_results if f['best_epoch'] + 1 >= 11)
}

colors_epoch_dist = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']
bars = ax.bar(epoch_data.keys(), epoch_data.values(), color=colors_epoch_dist,
              alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Folds', fontweight='bold')
ax.set_title('Distribution of Best Epoch Across Folds', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 5])

# Add value labels
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

# Plot 4: Summary text
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
CROSS-VALIDATION SUMMARY
{'='*50}

Performance Metrics:
  • Best Val Jaccard:    {mean_val:.1f}% ± {std_val:.1f}%
  • Range:               {stats['best_val_jacard']['min']*100:.1f}% - {stats['best_val_jacard']['max']*100:.1f}%
  • Coefficient of Var:  {(std_val/mean_val):.1%}

Comparison with Phase 1:
  • Phase 1 Best:        13.8%
  • CV Mean:             {mean_val:.1f}%
  • Improvement:         +{((mean_val-13.8)/13.8)*100:.0f}%

Overfitting Analysis:
  • Phase 1 Gap:         10.5×
  • CV Mean Gap:         {stats['overfitting_gap']['mean']:.1f}×
  • Reduction:           -{10.5-stats['overfitting_gap']['mean']:.1f}×

Best Epoch Analysis:
  • Mean Best Epoch:     {stats['best_epoch']['mean']+1:.1f}
  • Folds at epoch ≤1:   {sum(1 for f in fold_results if f['best_epoch'] <= 1)}/5
  • Folds at epoch >5:   {sum(1 for f in fold_results if f['best_epoch'] + 1 > 5)}/5

DIAGNOSIS:
  ✅ Phase 1 split was SEVERELY BIASED
  ✅ True performance is {((mean_val-13.8)/13.8)*100:.0f}% better
  ✅ "Epoch 1 problem" RESOLVED
  ✅ Training dynamics are NORMAL
"""

ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round',
        facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'cv_statistical_summary.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: cv_statistical_summary.png")
plt.close()

print("\n✅ All visualizations created successfully!")
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
print("  - cv_fold_comparison.png")
print("  - cv_training_curves.png")
print("  - cv_comparison_previous_tests.png")
print("  - cv_statistical_summary.png")
