#!/usr/bin/env python3
"""
Analyze Hyperparameter Search Results
======================================
Aggregates results from hyperparameter_search_20251013_154754/
Creates visualizations and comprehensive report
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
plt.rcParams['figure.figsize'] = (18, 14)
plt.rcParams['font.size'] = 11

results_dir = Path("hyperparameter_search_20251013_154754")

print("Loading hyperparameter search results...")

# Aggregate all results
all_results = []

for config_dir in sorted(results_dir.glob("resunet_*")):
    config_name = config_dir.name

    # Extract hyperparameters from directory name
    parts = config_name.split("_")
    architecture = parts[0]
    lr = float(parts[1].replace("lr", "").replace("e-0", "e-"))
    dropout = float(parts[2].replace("drop", ""))
    batch_size = int(parts[3].replace("bs", ""))

    fold_results = []

    for fold_dir in sorted(config_dir.glob("fold_*")):
        results_file = fold_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                fold_results.append(json.load(f))

    if fold_results:
        # Calculate statistics across folds
        best_jacards = [r['best_val_jacard'] for r in fold_results]
        best_epochs = [r['best_epoch'] for r in fold_results]
        overfitting_gaps = [r['overfitting_gap'] for r in fold_results]

        config_summary = {
            'architecture': architecture,
            'config_name': config_name,
            'learning_rate': lr,
            'dropout': dropout,
            'batch_size': batch_size,
            'n_folds': len(fold_results),
            'mean_best_jacard': float(np.mean(best_jacards)),
            'std_best_jacard': float(np.std(best_jacards)),
            'min_best_jacard': float(np.min(best_jacards)),
            'max_best_jacard': float(np.max(best_jacards)),
            'mean_best_epoch': float(np.mean(best_epochs)),
            'mean_overfitting_gap': float(np.mean(overfitting_gaps)),
            'fold_results': fold_results
        }

        all_results.append(config_summary)

print(f"Loaded {len(all_results)} configurations")

# Convert to DataFrame for easier analysis
df = pd.DataFrame([{
    'config_name': r['config_name'],
    'learning_rate': r['learning_rate'],
    'dropout': r['dropout'],
    'batch_size': r['batch_size'],
    'mean_jacard': r['mean_best_jacard'],
    'std_jacard': r['std_best_jacard'],
    'mean_epoch': r['mean_best_epoch'],
    'mean_gap': r['mean_overfitting_gap']
} for r in all_results])

# Find best configuration
best_idx = df['mean_jacard'].idxmax()
best_config = all_results[best_idx]

print(f"\nBest configuration: {best_config['config_name']}")
print(f"Performance: {best_config['mean_best_jacard']:.4f} ± {best_config['std_best_jacard']:.4f}")

# ============================================================================
# Figure 1: Hyperparameter Effects
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Hyperparameter Effects on ResUNet Performance', fontsize=18, fontweight='bold', y=0.995)

# 1. Learning Rate Effect
ax = axes[0, 0]
lr_groups = df.groupby('learning_rate')['mean_jacard'].agg(['mean', 'std', 'count'])
lr_values = lr_groups.index.tolist()
lr_means = lr_groups['mean'].tolist()
lr_stds = lr_groups['std'].tolist()

x_pos = np.arange(len(lr_values))
bars = ax.bar(x_pos, lr_means, yerr=lr_stds, capsize=8, alpha=0.7,
              color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', linewidth=1.5)

ax.set_xlabel('Learning Rate', fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Best Val Jaccard', fontweight='bold', fontsize=12)
ax.set_title('(A) Learning Rate Effect', fontweight='bold', fontsize=14, pad=10)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{lr:.0e}' for lr in lr_values])
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.6994, color='red', linestyle='--', linewidth=2, label='U-Net baseline (69.94%)')
ax.axhline(y=0.3995, color='orange', linestyle='--', linewidth=2, label='ResUNet baseline (39.95%)')
ax.legend(fontsize=9)

# Add value labels
for bar, mean in zip(bars, lr_means):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
           f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. Dropout Effect
ax = axes[0, 1]
dropout_groups = df.groupby('dropout')['mean_jacard'].agg(['mean', 'std', 'count'])
dropout_values = dropout_groups.index.tolist()
dropout_means = dropout_groups['mean'].tolist()
dropout_stds = dropout_groups['std'].tolist()

x_pos = np.arange(len(dropout_values))
bars = ax.bar(x_pos, dropout_means, yerr=dropout_stds, capsize=8, alpha=0.7,
              color=['#d62728', '#9467bd', '#8c564b'], edgecolor='black', linewidth=1.5)

ax.set_xlabel('Dropout', fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Best Val Jaccard', fontweight='bold', fontsize=12)
ax.set_title('(B) Dropout Effect', fontweight='bold', fontsize=14, pad=10)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{d:.1f}' for d in dropout_values])
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.6994, color='red', linestyle='--', linewidth=2, label='U-Net baseline')
ax.axhline(y=0.3995, color='orange', linestyle='--', linewidth=2, label='ResUNet baseline')
ax.legend(fontsize=9)

for bar, mean in zip(bars, dropout_means):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
           f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Batch Size Effect
ax = axes[0, 2]
bs_groups = df.groupby('batch_size')['mean_jacard'].agg(['mean', 'std', 'count'])
bs_values = bs_groups.index.tolist()
bs_means = bs_groups['mean'].tolist()
bs_stds = bs_groups['std'].tolist()

x_pos = np.arange(len(bs_values))
bars = ax.bar(x_pos, bs_means, yerr=bs_stds, capsize=8, alpha=0.7,
              color=['#e377c2', '#7f7f7f'], edgecolor='black', linewidth=1.5)

ax.set_xlabel('Batch Size', fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Best Val Jaccard', fontweight='bold', fontsize=12)
ax.set_title('(C) Batch Size Effect', fontweight='bold', fontsize=14, pad=10)
ax.set_xticks(x_pos)
ax.set_xticklabels([str(bs) for bs in bs_values])
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.6994, color='red', linestyle='--', linewidth=2, label='U-Net baseline')
ax.axhline(y=0.3995, color='orange', linestyle='--', linewidth=2, label='ResUNet baseline')
ax.legend(fontsize=9)

for bar, mean in zip(bars, bs_means):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
           f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. Learning Rate vs Best Epoch
ax = axes[1, 0]
for lr in df['learning_rate'].unique():
    lr_data = df[df['learning_rate'] == lr]
    ax.scatter(lr_data['mean_epoch'], lr_data['mean_jacard'],
              label=f'LR={lr:.0e}', s=100, alpha=0.7)

ax.set_xlabel('Mean Best Epoch', fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Best Val Jaccard', fontweight='bold', fontsize=12)
ax.set_title('(D) Convergence Speed vs Performance', fontweight='bold', fontsize=14, pad=10)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.axhline(y=0.6994, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=0.3995, color='orange', linestyle='--', linewidth=1, alpha=0.5)

# 5. Overfitting Gap Analysis
ax = axes[1, 1]
for lr in df['learning_rate'].unique():
    lr_data = df[df['learning_rate'] == lr]
    ax.scatter(lr_data['mean_gap'], lr_data['mean_jacard'],
              label=f'LR={lr:.0e}', s=100, alpha=0.7)

ax.set_xlabel('Mean Overfitting Gap (Train/Val)', fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Best Val Jaccard', fontweight='bold', fontsize=12)
ax.set_title('(E) Overfitting vs Performance', fontweight='bold', fontsize=14, pad=10)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.axhline(y=0.6994, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=0.3995, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=2.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Target gap (2×)')

# 6. Top 5 Configurations
ax = axes[1, 2]
ax.axis('off')

top_5 = df.nlargest(5, 'mean_jacard')
table_data = [['Rank', 'Config', 'Jaccard', 'LR', 'Drop', 'BS']]
table_data.append(['─'*4, '─'*20, '─'*8, '─'*6, '─'*4, '─'*3])

for idx, (rank, row) in enumerate(top_5.iterrows(), 1):
    lr_str = f"{row['learning_rate']:.0e}"
    config_short = f"LR={lr_str} D={row['dropout']:.1f} BS={row['batch_size']}"
    table_data.append([
        str(rank),
        config_short,
        f"{row['mean_jacard']:.4f}",
        lr_str,
        f"{row['dropout']:.1f}",
        str(int(row['batch_size']))
    ])

table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                colWidths=[0.08, 0.35, 0.15, 0.12, 0.12, 0.08])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.8)

# Style header
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#404040')
    cell.set_text_props(weight='bold', color='white')

# Style data rows
for i in range(2, len(table_data)):
    for j in range(6):
        cell = table[(i, j)]
        if i == 2:  # Best config
            cell.set_facecolor('#90EE90')
            cell.set_alpha(0.7)
        else:
            cell.set_facecolor('#f0f0f0')
            cell.set_alpha(0.5)

ax.set_title('(F) Top 5 Configurations', fontweight='bold', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig(results_dir / 'hyperparam_effects_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: hyperparam_effects_analysis.png")
plt.close()

# ============================================================================
# Figure 2: Heatmaps
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Hyperparameter Interaction Heatmaps', fontsize=18, fontweight='bold')

# 1. LR vs Dropout (BS=4)
ax = axes[0]
pivot = df[df['batch_size'] == 4].pivot_table(
    values='mean_jacard', index='dropout', columns='learning_rate'
)
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', center=0.3,
            ax=ax, cbar_kws={'label': 'Mean Jaccard'}, vmin=0, vmax=0.7)
ax.set_title('LR vs Dropout (Batch Size = 4)', fontweight='bold', fontsize=14)
ax.set_xlabel('Learning Rate', fontweight='bold')
ax.set_ylabel('Dropout', fontweight='bold')
ax.set_xticklabels([f'{float(col):.0e}' for col in pivot.columns])

# 2. LR vs Dropout (BS=8)
ax = axes[1]
pivot = df[df['batch_size'] == 8].pivot_table(
    values='mean_jacard', index='dropout', columns='learning_rate'
)
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', center=0.3,
            ax=ax, cbar_kws={'label': 'Mean Jaccard'}, vmin=0, vmax=0.7)
ax.set_title('LR vs Dropout (Batch Size = 8)', fontweight='bold', fontsize=14)
ax.set_xlabel('Learning Rate', fontweight='bold')
ax.set_ylabel('Dropout', fontweight='bold')
ax.set_xticklabels([f'{float(col):.0e}' for col in pivot.columns])

# 3. LR vs Batch Size (averaged over dropout)
ax = axes[2]
pivot = df.groupby(['learning_rate', 'batch_size'])['mean_jacard'].mean().unstack()
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', center=0.3,
            ax=ax, cbar_kws={'label': 'Mean Jaccard'}, vmin=0, vmax=0.7)
ax.set_title('LR vs Batch Size (avg over dropout)', fontweight='bold', fontsize=14)
ax.set_xlabel('Batch Size', fontweight='bold')
ax.set_ylabel('Learning Rate', fontweight='bold')
ax.set_yticklabels([f'{float(idx):.0e}' for idx in pivot.index])

plt.tight_layout()
plt.savefig(results_dir / 'hyperparam_heatmaps.png', dpi=300, bbox_inches='tight')
print("✓ Saved: hyperparam_heatmaps.png")
plt.close()

# ============================================================================
# Figure 3: Comparison with Baselines
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comparison with Baseline Architectures', fontsize=18, fontweight='bold', y=0.995)

# 1. All configs vs baselines
ax = axes[0, 0]
x = np.arange(len(all_results))
means = [r['mean_best_jacard'] for r in all_results]
stds = [r['std_best_jacard'] for r in all_results]

ax.bar(x, means, yerr=stds, alpha=0.6, capsize=3, color='steelblue', edgecolor='black')
ax.axhline(y=0.6994, color='red', linestyle='--', linewidth=2, label='U-Net (69.94%)')
ax.axhline(y=0.3995, color='orange', linestyle='--', linewidth=2, label='ResUNet baseline (39.95%)')
ax.axhline(y=best_config['mean_best_jacard'], color='green', linestyle='--', linewidth=2,
           label=f"Best found ({best_config['mean_best_jacard']:.2%})")

ax.set_xlabel('Configuration Index', fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Best Val Jaccard', fontweight='bold', fontsize=12)
ax.set_title('(A) All Configurations vs Baselines', fontweight='bold', fontsize=14, pad=10)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.75)

# 2. Distribution comparison
ax = axes[0, 1]
data_for_box = [
    [0.6994],  # U-Net (single point)
    [0.3995],  # ResUNet baseline (single point)
    means      # All search results
]
labels = ['U-Net\n(baseline)', 'ResUNet\n(baseline)', f'Search Results\n(n={len(means)})']

bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

ax.set_ylabel('Best Val Jaccard', fontweight='bold', fontsize=12)
ax.set_title('(B) Performance Distribution', fontweight='bold', fontsize=14, pad=10)
ax.grid(axis='y', alpha=0.3)

# 3. Best config fold-by-fold
ax = axes[1, 0]
best_fold_results = best_config['fold_results']
fold_jacards = [r['best_val_jacard'] for r in best_fold_results]
fold_nums = list(range(1, len(fold_jacards) + 1))

ax.bar(fold_nums, fold_jacards, alpha=0.7, color='green', edgecolor='black', linewidth=1.5)
ax.axhline(y=best_config['mean_best_jacard'], color='darkgreen', linestyle='--',
           linewidth=2, label=f"Mean: {best_config['mean_best_jacard']:.4f}")
ax.axhline(y=0.6994, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='U-Net')
ax.axhline(y=0.3995, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='ResUNet baseline')

ax.set_xlabel('Fold', fontweight='bold', fontsize=12)
ax.set_ylabel('Best Val Jaccard', fontweight='bold', fontsize=12)
ax.set_title(f'(C) Best Config Performance by Fold\n{best_config["config_name"]}',
            fontweight='bold', fontsize=14, pad=10)
ax.set_xticks(fold_nums)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# 4. Summary table
ax = axes[1, 1]
ax.axis('off')

table_data = [
    ['Metric', 'U-Net', 'ResUNet\nBaseline', 'ResUNet\nOptimized'],
    ['─'*15, '─'*10, '─'*12, '─'*12],
    ['Best Jaccard', '69.94%', '39.95%', f"{best_config['mean_best_jacard']*100:.2f}%"],
    ['Std Dev', '5.02%', '6.79%', f"{best_config['std_best_jacard']*100:.2f}%"],
    ['Best Epoch', '9.8', '2.6', f"{best_config['mean_best_epoch']:.1f}"],
    ['Overfitting Gap', '1.98×', '3.11×', f"{best_config['mean_overfitting_gap']:.2f}×"],
    ['Learning Rate', '5e-5', '5e-5', f"{best_config['learning_rate']:.0e}"],
    ['Dropout', '0.3', '0.3', f"{best_config['dropout']:.1f}"],
    ['Batch Size', '4', '4', f"{best_config['batch_size']}"],
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                colWidths=[0.35, 0.22, 0.22, 0.22])
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
    cell = table[(i, 0)]
    cell.set_facecolor('#f0f0f0')
    cell.set_text_props(weight='bold')

    cell = table[(i, 1)]
    cell.set_facecolor('#90EE90')
    cell.set_alpha(0.5)

    cell = table[(i, 2)]
    cell.set_facecolor('#FFB6C1')
    cell.set_alpha(0.5)

    cell = table[(i, 3)]
    # Best optimized cell
    if i == 2:  # Jaccard row
        jaccard = float(table_data[i][3].strip('%'))
        if jaccard > 39.95:
            cell.set_facecolor('#FFFF99')
        else:
            cell.set_facecolor('#FFB6C1')
    else:
        cell.set_facecolor('#FFFF99')
    cell.set_alpha(0.5)

ax.set_title('(D) Summary Comparison', fontweight='bold', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig(results_dir / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: baseline_comparison.png")
plt.close()

# Save aggregated results
summary = {
    'n_configurations': len(all_results),
    'best_config': best_config,
    'all_configs': all_results,
    'hyperparameter_effects': {
        'learning_rate': lr_groups.to_dict(),
        'dropout': dropout_groups.to_dict(),
        'batch_size': bs_groups.to_dict()
    },
    'baselines': {
        'unet': 0.6994,
        'resunet_baseline': 0.3995
    }
}

with open(results_dir / 'hyperparameter_search_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✅ Analysis complete!")
print(f"Generated files in {results_dir}:")
print("  - hyperparam_effects_analysis.png")
print("  - hyperparam_heatmaps.png")
print("  - baseline_comparison.png")
print("  - hyperparameter_search_summary.json")

print(f"\n🏆 BEST CONFIGURATION:")
print(f"  Config: {best_config['config_name']}")
print(f"  Performance: {best_config['mean_best_jacard']:.4f} ± {best_config['std_best_jacard']:.4f}")
print(f"  vs U-Net (69.94%): {((best_config['mean_best_jacard'] - 0.6994) / 0.6994 * 100):+.1f}%")
print(f"  vs ResUNet baseline (39.95%): {((best_config['mean_best_jacard'] - 0.3995) / 0.3995 * 100):+.1f}%")
