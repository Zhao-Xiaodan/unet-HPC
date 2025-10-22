#!/usr/bin/env python3
"""
Comprehensive analysis of PyTorch comparison experiments.

Analyzes three training approaches:
1. No Augmentation + BinaryFocalLoss
2. With Augmentation + BinaryFocalLoss
3. With Augmentation + AdaptiveBGDiceLoss

Generates visualizations and statistics for the report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Define directories
DIR_NO_AUG = "pytorch_comparison_no_aug_20251021_121918"
DIR_WITH_AUG = "pytorch_comparison_with_aug_20251021_122018"
DIR_ADAPTIVE = "pytorch_comparison_adaptive_loss_20251021_121920"

OUTPUT_DIR = Path("pytorch_comparison_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
print("Loading data...")
df_no_aug = pd.read_csv(f"{DIR_NO_AUG}/all_results.csv")
df_with_aug = pd.read_csv(f"{DIR_WITH_AUG}/all_results.csv")
df_adaptive = pd.read_csv(f"{DIR_ADAPTIVE}/all_results.csv")

# Add experiment labels
df_no_aug['experiment'] = 'No Aug + BinaryFocal'
df_with_aug['experiment'] = 'With Aug + BinaryFocal'
df_adaptive['experiment'] = 'With Aug + AdaptiveLoss'

# Combine all data
df_all = pd.concat([df_no_aug, df_with_aug, df_adaptive], ignore_index=True)

# Clean architecture names for display
arch_map = {
    'unet': 'UNet',
    'attention_unet': 'Attention UNet',
    'attention_resunet': 'Attention ResUNet'
}
df_all['Architecture'] = df_all['architecture'].map(arch_map)

print(f"Total models trained: {len(df_all)}")
print(f"Models per experiment: {len(df_all) // 3}")
print()

# ============================================================================
# Figure 1: Overall Performance Comparison (Box Plot)
# ============================================================================
print("Creating Figure 1: Overall performance comparison...")

fig, ax = plt.subplots(figsize=(12, 6))

# Create box plot
df_plot = df_all.copy()
df_plot['Experiment'] = df_plot['experiment']

# Filter out catastrophic failures (IoU < 0.1)
df_plot_filtered = df_plot[df_plot['best_val_iou'] >= 0.1].copy()

sns.boxplot(
    data=df_plot_filtered,
    x='Architecture',
    y='best_val_iou',
    hue='Experiment',
    ax=ax,
    palette='Set2'
)

ax.set_ylabel('Validation IoU', fontsize=12, fontweight='bold')
ax.set_xlabel('Architecture', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison Across Experiments and Architectures',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(title='Experiment', loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig1_overall_performance.png', bbox_inches='tight')
plt.close()

print(f"  Saved: {OUTPUT_DIR / 'fig1_overall_performance.png'}")

# ============================================================================
# Figure 2: Best Model Per Architecture Per Experiment
# ============================================================================
print("Creating Figure 2: Best models comparison...")

# Find best model for each architecture-experiment combination
best_models = df_all.groupby(['Architecture', 'experiment'])['best_val_iou'].max().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))

# Create grouped bar plot
x = np.arange(len(arch_map))
width = 0.25

experiments = df_all['experiment'].unique()
colors = sns.color_palette('Set2', n_colors=len(experiments))

for i, exp in enumerate(experiments):
    exp_data = best_models[best_models['experiment'] == exp]
    values = [exp_data[exp_data['Architecture'] == arch]['best_val_iou'].values[0]
              if len(exp_data[exp_data['Architecture'] == arch]) > 0 else 0
              for arch in arch_map.values()]

    ax.bar(x + i*width, values, width, label=exp, color=colors[i], alpha=0.8)

ax.set_ylabel('Best Validation IoU', fontsize=12, fontweight='bold')
ax.set_xlabel('Architecture', fontsize=12, fontweight='bold')
ax.set_title('Best Model Performance by Architecture and Experiment',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(arch_map.values())
ax.legend(title='Experiment', fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 0.7])

# Add value labels on bars
for i, exp in enumerate(experiments):
    exp_data = best_models[best_models['experiment'] == exp]
    for j, arch in enumerate(arch_map.values()):
        value = exp_data[exp_data['Architecture'] == arch]['best_val_iou'].values[0] \
                if len(exp_data[exp_data['Architecture'] == arch]) > 0 else 0
        ax.text(j + i*width, value + 0.01, f'{value:.3f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig2_best_models.png', bbox_inches='tight')
plt.close()

print(f"  Saved: {OUTPUT_DIR / 'fig2_best_models.png'}")

# ============================================================================
# Figure 3: Hyperparameter Sensitivity (Heatmap)
# ============================================================================
print("Creating Figure 3: Hyperparameter sensitivity...")

fig, axes = plt.subplots(3, 3, figsize=(16, 14))

for exp_idx, exp in enumerate(experiments):
    for arch_idx, arch in enumerate(arch_map.values()):
        ax = axes[exp_idx, arch_idx]

        # Filter data
        data = df_all[(df_all['experiment'] == exp) & (df_all['Architecture'] == arch)]

        # Create pivot table
        pivot = data.pivot_table(
            values='best_val_iou',
            index='dropout',
            columns='learning_rate',
            aggfunc='mean'
        )

        # Plot heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                   vmin=0.0, vmax=0.7, ax=ax, cbar_kws={'label': 'Val IoU'})

        ax.set_title(f'{exp}\n{arch}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Learning Rate', fontsize=10)
        ax.set_ylabel('Dropout', fontsize=10)

plt.suptitle('Hyperparameter Sensitivity Across Experiments and Architectures',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig3_hyperparameter_sensitivity.png', bbox_inches='tight')
plt.close()

print(f"  Saved: {OUTPUT_DIR / 'fig3_hyperparameter_sensitivity.png'}")

# ============================================================================
# Figure 4: Training Stability (Coefficient of Variation)
# ============================================================================
print("Creating Figure 4: Training stability analysis...")

# Calculate CV for each architecture-experiment combination
stability_data = []

for exp in experiments:
    for arch in arch_map.values():
        data = df_all[(df_all['experiment'] == exp) & (df_all['Architecture'] == arch)]

        # Filter out catastrophic failures
        data_filtered = data[data['best_val_iou'] >= 0.1]

        mean_iou = data_filtered['best_val_iou'].mean()
        std_iou = data_filtered['best_val_iou'].std()
        cv = (std_iou / mean_iou) * 100 if mean_iou > 0 else 0

        stability_data.append({
            'Architecture': arch,
            'Experiment': exp,
            'Mean IoU': mean_iou,
            'Std IoU': std_iou,
            'CV (%)': cv,
            'N_models': len(data_filtered),
            'N_failures': len(data) - len(data_filtered)
        })

df_stability = pd.DataFrame(stability_data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Coefficient of Variation
x = np.arange(len(arch_map))
width = 0.25

for i, exp in enumerate(experiments):
    exp_data = df_stability[df_stability['Experiment'] == exp]
    values = [exp_data[exp_data['Architecture'] == arch]['CV (%)'].values[0]
              for arch in arch_map.values()]

    ax1.bar(x + i*width, values, width, label=exp, alpha=0.8)

ax1.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Architecture', fontsize=12, fontweight='bold')
ax1.set_title('Training Stability (Lower is Better)', fontsize=13, fontweight='bold')
ax1.set_xticks(x + width)
ax1.set_xticklabels(arch_map.values())
ax1.legend(title='Experiment', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Failure Count
failure_data = df_stability.pivot(index='Architecture', columns='Experiment', values='N_failures')
failure_data.plot(kind='bar', ax=ax2, alpha=0.8)

ax2.set_ylabel('Number of Failed Models (IoU < 0.1)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Architecture', fontsize=12, fontweight='bold')
ax2.set_title('Training Failures', fontsize=13, fontweight='bold')
ax2.legend(title='Experiment', fontsize=9)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig4_stability_analysis.png', bbox_inches='tight')
plt.close()

print(f"  Saved: {OUTPUT_DIR / 'fig4_stability_analysis.png'}")

# ============================================================================
# Figure 5: Impact of Augmentation (Paired Comparison)
# ============================================================================
print("Creating Figure 5: Augmentation impact...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for arch_idx, arch in enumerate(arch_map.values()):
    ax = axes[arch_idx]

    # Get data for no aug vs with aug (both using BinaryFocal)
    no_aug = df_all[(df_all['experiment'] == 'No Aug + BinaryFocal') &
                    (df_all['Architecture'] == arch)].sort_values(
                        ['n_filters', 'dropout', 'learning_rate'])
    with_aug = df_all[(df_all['experiment'] == 'With Aug + BinaryFocal') &
                      (df_all['Architecture'] == arch)].sort_values(
                        ['n_filters', 'dropout', 'learning_rate'])

    # Scatter plot
    ax.scatter(no_aug['best_val_iou'], with_aug['best_val_iou'], alpha=0.6, s=50)

    # Add diagonal line
    max_val = max(no_aug['best_val_iou'].max(), with_aug['best_val_iou'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='No change')

    ax.set_xlabel('No Augmentation (IoU)', fontsize=11, fontweight='bold')
    ax.set_ylabel('With Augmentation (IoU)', fontsize=11, fontweight='bold')
    ax.set_title(f'{arch}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    # Add statistics (reset index for element-wise comparison)
    no_aug_vals = no_aug['best_val_iou'].values
    with_aug_vals = with_aug['best_val_iou'].values
    mean_diff = (with_aug_vals - no_aug_vals).mean()
    n_improved = (with_aug_vals > no_aug_vals).sum()
    n_total = len(with_aug_vals)

    ax.text(0.05, 0.95, f'Δ IoU: {mean_diff:+.4f}\n{n_improved}/{n_total} improved',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Impact of Augmentation on Performance (BinaryFocal Loss)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig5_augmentation_impact.png', bbox_inches='tight')
plt.close()

print(f"  Saved: {OUTPUT_DIR / 'fig5_augmentation_impact.png'}")

# ============================================================================
# Figure 6: Impact of Loss Function (Paired Comparison)
# ============================================================================
print("Creating Figure 6: Loss function impact...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for arch_idx, arch in enumerate(arch_map.values()):
    ax = axes[arch_idx]

    # Get data for BinaryFocal vs AdaptiveLoss (both with augmentation)
    binary_focal = df_all[(df_all['experiment'] == 'With Aug + BinaryFocal') &
                          (df_all['Architecture'] == arch)].sort_values(
                            ['n_filters', 'dropout', 'learning_rate'])
    adaptive = df_all[(df_all['experiment'] == 'With Aug + AdaptiveLoss') &
                      (df_all['Architecture'] == arch)].sort_values(
                        ['n_filters', 'dropout', 'learning_rate'])

    # Scatter plot
    ax.scatter(binary_focal['best_val_iou'], adaptive['best_val_iou'],
               alpha=0.6, s=50, color='coral')

    # Add diagonal line
    max_val = max(binary_focal['best_val_iou'].max(), adaptive['best_val_iou'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='No change')

    ax.set_xlabel('BinaryFocal Loss (IoU)', fontsize=11, fontweight='bold')
    ax.set_ylabel('AdaptiveBGDice Loss (IoU)', fontsize=11, fontweight='bold')
    ax.set_title(f'{arch}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    # Add statistics (reset index for element-wise comparison)
    binary_vals = binary_focal['best_val_iou'].values
    adaptive_vals = adaptive['best_val_iou'].values
    mean_diff = (adaptive_vals - binary_vals).mean()
    n_improved = (adaptive_vals > binary_vals).sum()
    n_total = len(adaptive_vals)

    ax.text(0.05, 0.95, f'Δ IoU: {mean_diff:+.4f}\n{n_improved}/{n_total} improved',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('Impact of Loss Function on Performance (With Augmentation)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig6_loss_function_impact.png', bbox_inches='tight')
plt.close()

print(f"  Saved: {OUTPUT_DIR / 'fig6_loss_function_impact.png'}")

# ============================================================================
# Generate Summary Statistics Table
# ============================================================================
print("\nGenerating summary statistics...")

summary_stats = []

for exp in experiments:
    for arch in arch_map.values():
        data = df_all[(df_all['experiment'] == exp) & (df_all['Architecture'] == arch)]
        data_filtered = data[data['best_val_iou'] >= 0.1]  # Exclude failures

        best_iou = data['best_val_iou'].max()
        mean_iou = data_filtered['best_val_iou'].mean()
        std_iou = data_filtered['best_val_iou'].std()
        median_iou = data_filtered['best_val_iou'].median()

        # Best hyperparameters
        best_model = data.loc[data['best_val_iou'].idxmax()]

        summary_stats.append({
            'Experiment': exp,
            'Architecture': arch,
            'Best_IoU': best_iou,
            'Mean_IoU': mean_iou,
            'Median_IoU': median_iou,
            'Std_IoU': std_iou,
            'N_Models': len(data_filtered),
            'N_Failures': len(data) - len(data_filtered),
            'Best_n_filters': int(best_model['n_filters']),
            'Best_dropout': best_model['dropout'],
            'Best_lr': best_model['learning_rate']
        })

df_summary = pd.DataFrame(summary_stats)

# Save to CSV
df_summary.to_csv(OUTPUT_DIR / 'summary_statistics.csv', index=False, float_format='%.4f')
print(f"  Saved: {OUTPUT_DIR / 'summary_statistics.csv'}")

# Save detailed comparison table
comparison_table = df_summary.pivot_table(
    index='Architecture',
    columns='Experiment',
    values='Best_IoU'
)
comparison_table.to_csv(OUTPUT_DIR / 'best_iou_comparison.csv', float_format='%.4f')
print(f"  Saved: {OUTPUT_DIR / 'best_iou_comparison.csv'}")

# ============================================================================
# Print Summary to Console
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nBest IoU by Architecture and Experiment:")
print(comparison_table.to_string())

print("\n\nOverall Best Models:")
for arch in arch_map.values():
    arch_data = df_summary[df_summary['Architecture'] == arch]
    best_exp = arch_data.loc[arch_data['Best_IoU'].idxmax()]

    print(f"\n{arch}:")
    print(f"  Best IoU: {best_exp['Best_IoU']:.4f}")
    print(f"  Experiment: {best_exp['Experiment']}")
    print(f"  Hyperparameters: n_filters={best_exp['Best_n_filters']}, "
          f"dropout={best_exp['Best_dropout']}, lr={best_exp['Best_lr']}")

print("\n\nTraining Stability (CV %):")
print(df_stability.pivot_table(
    index='Architecture',
    columns='Experiment',
    values='CV (%)'
).to_string(float_format='%.2f'))

print("\n\nNumber of Failed Models (IoU < 0.1):")
print(df_stability.pivot_table(
    index='Architecture',
    columns='Experiment',
    values='N_failures'
).to_string(float_format='%.0f'))

print("\n" + "="*80)
print("Analysis complete! All figures saved to:", OUTPUT_DIR)
print("="*80)
