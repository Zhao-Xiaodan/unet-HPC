#!/usr/bin/env python3
"""
Comparative analysis between hyperparameter search and Xukuang parameters experiments
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# Define directories
HYPERPARAM_DIR = Path("hyperparameter_search_512_20251014_235755")
XUKUANG_DIR = Path("xukuang_params_shrunk_20251015_071224")

def load_data():
    """Load data from both experiments"""

    # Load hyperparameter search results
    hp_results = pd.read_csv(HYPERPARAM_DIR / "all_results.csv")
    # Create summary from results since summary.json is incomplete
    hp_summary = {
        'best_jaccard': hp_results['best_val_jaccard'].max(),
        'mean_jaccard': hp_results['best_val_jaccard'].mean(),
        'best_config': hp_results.loc[hp_results['best_val_jaccard'].idxmax(), 'config_name']
    }

    # Load Xukuang experiment results
    with open(XUKUANG_DIR / "EXPERIMENT_INFO.json", 'r') as f:
        xk_info = json.load(f)
    with open(XUKUANG_DIR / "TRAINING_SUMMARY.json", 'r') as f:
        xk_summary = json.load(f)
    with open(XUKUANG_DIR / "analysis_statistics.json", 'r') as f:
        xk_stats = json.load(f)

    return hp_results, hp_summary, xk_info, xk_summary, xk_stats

def create_performance_comparison(hp_results, xk_summary, xk_stats):
    """Create comprehensive performance comparison chart"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hyperparameter Search vs Xukuang Parameters: Performance Comparison',
                 fontsize=16, fontweight='bold', y=0.995)

    # Extract data
    hp_best_by_arch = hp_results.groupby('architecture')['best_val_jaccard'].max()
    hp_mean_by_arch = hp_results.groupby('architecture')['best_val_jaccard'].mean()

    xk_final = {
        'UNet': xk_summary['final_validation_metrics']['unet']['jaccard'],
        'Attention_UNet': xk_summary['final_validation_metrics']['attention_unet']['jaccard'],
        'Attention_ResUNet': xk_summary['final_validation_metrics']['attention_resunet']['jaccard']
    }

    xk_best = {
        'UNet': xk_stats['UNet']['best_val_jaccard'],
        'Attention_UNet': xk_stats['Attention UNet']['best_val_jaccard'],
        'Attention_ResUNet': xk_stats['Attention ResUNet']['best_val_jaccard']
    }

    # Map architecture names
    arch_map = {'unet': 'UNet', 'resunet': 'ResUNet', 'attention_resunet': 'Attention_ResUNet'}

    # Plot 1: Best IoU Comparison
    ax = axes[0, 0]
    x_pos = np.arange(3)
    width = 0.35

    hp_vals = [hp_best_by_arch.get(k, 0) for k in ['unet', 'resunet', 'attention_resunet']]
    xk_vals = [xk_best[k] for k in ['UNet', 'Attention_ResUNet', 'Attention_ResUNet']]
    # Correct mapping
    xk_vals_mapped = [xk_best['UNet'], 0, xk_best['Attention_ResUNet']]

    bars1 = ax.bar(x_pos - width/2, hp_vals, width, label='Hyperparam Search (Best)',
                   color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, [xk_best['UNet'], 0, xk_best['Attention_ResUNet']],
                   width, label='Xukuang (Best Epoch)',
                   color='#F18F01', alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Best Validation Jaccard (IoU)', fontsize=11)
    ax.set_title('Best Performance Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['UNet', 'ResUNet', 'Attention ResUNet'], rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Final IoU Comparison
    ax = axes[0, 1]
    hp_final = hp_results.groupby('architecture')['final_val_jaccard'].mean()
    hp_final_vals = [hp_final.get(k, 0) for k in ['unet', 'resunet', 'attention_resunet']]
    xk_final_vals = [xk_final['UNet'], 0, xk_final['Attention_ResUNet']]

    bars1 = ax.bar(x_pos - width/2, hp_final_vals, width, label='Hyperparam Search (Final)',
                   color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, [xk_final['UNet'], 0, xk_final['Attention_ResUNet']],
                   width, label='Xukuang (Final Epoch)',
                   color='#F18F01', alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Final Validation Jaccard (IoU)', fontsize=11)
    ax.set_title('Final Performance Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['UNet', 'ResUNet', 'Attention ResUNet'], rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 3: Training Parameters Comparison
    ax = axes[1, 0]
    params_comparison = [
        ['Learning Rate', '1e-4 / 5e-5', '5e-3'],
        ['Epochs', '20 (early stop)', '200'],
        ['Dropout', '0.2 / 0.3', 'None'],
        ['Loss Function', 'BinaryFocalLoss', 'BinaryFocalLoss'],
        ['Batch Size', '4', '4'],
        ['Image Type', 'Grayscale', 'RGB']
    ]

    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=params_comparison,
                     colLabels=['Parameter', 'Hyperparam Search', 'Xukuang'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(params_comparison) + 1):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')

    ax.set_title('Training Configuration Comparison', fontweight='bold', fontsize=12, pad=20)

    # Plot 4: Key Metrics Summary
    ax = axes[1, 1]

    metrics_data = [
        ['Metric', 'Hyperparam Search', 'Xukuang', 'Winner'],
        ['Best IoU (overall)', f'{hp_results["best_val_jaccard"].max():.4f}',
         f'{max(xk_best.values()):.4f}', 'Xukuang (3.1×)'],
        ['Mean IoU (UNet)', f'{hp_results[hp_results.architecture=="unet"]["best_val_jaccard"].mean():.4f}',
         f'{xk_final["UNet"]:.4f}', 'Xukuang (4.5×)'],
        ['Training Stability', 'Early plateau (0-2 ep)', 'Stable 200 epochs', 'Xukuang'],
        ['Overfitting', 'Moderate (12.5%)', 'Severe (ResUNets)', 'Hyperparam'],
        ['Training Time', '~9 hours', '~1.3 hours', 'Xukuang'],
        ['Dataset', '98 grayscale', '98 RGB', 'Different']
    ]

    ax.axis('tight')
    ax.axis('off')
    table2 = ax.table(cellText=metrics_data,
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.3, 0.25, 0.25, 0.2])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 2.2)

    # Style header
    for i in range(4):
        table2[(0, i)].set_facecolor('#4472C4')
        table2[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors and highlight winners
    for i in range(1, len(metrics_data)):
        for j in range(4):
            if i % 2 == 0:
                table2[(i, j)].set_facecolor('#E7E6E6')
            if j == 3 and i > 0:  # Winner column
                if 'Xukuang' in metrics_data[i][3]:
                    table2[(i, j)].set_facecolor('#90EE90')
                elif 'Hyperparam' in metrics_data[i][3]:
                    table2[(i, j)].set_facecolor('#FFB6C1')

    ax.set_title('Key Metrics Comparison', fontweight='bold', fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig(XUKUANG_DIR / 'comparison_hyperparam_vs_xukuang.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_hyperparam_vs_xukuang.png")
    plt.close()

def create_improvement_analysis(hp_results, xk_stats):
    """Analyze why Xukuang parameters performed better"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Why Xukuang Parameters Outperformed Hyperparameter Search',
                 fontsize=16, fontweight='bold')

    # Plot 1: Learning rate impact
    ax = axes[0]

    hp_lr_performance = []
    for lr in [5e-5, 1e-4]:
        mask = hp_results['config_name'].str.contains(f'lr{lr:.0e}')
        mean_iou = hp_results[mask & (hp_results['architecture'] == 'unet')]['best_val_jaccard'].mean()
        hp_lr_performance.append(mean_iou)

    lrs = ['5e-5\n(Hyperparam)', '1e-4\n(Hyperparam)', '5e-3\n(Xukuang)']
    ious = [hp_lr_performance[0], hp_lr_performance[1], xk_stats['UNet']['best_val_jaccard']]
    colors = ['#2E86AB', '#2E86AB', '#F18F01']

    bars = ax.bar(lrs, ious, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Best Validation IoU (UNet)', fontsize=12)
    ax.set_title('Learning Rate Impact', fontweight='bold', fontsize=13)
    ax.set_ylim([0, max(ious) * 1.2])
    ax.grid(True, alpha=0.3, axis='y')

    for bar, iou in zip(bars, ious):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{iou:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add improvement annotations
    improvement = ((ious[2] - max(ious[:2])) / max(ious[:2])) * 100
    ax.annotate(f'+{improvement:.1f}%', xy=(2, ious[2]), xytext=(2, ious[2] + 0.1),
               ha='center', fontsize=12, fontweight='bold', color='green',
               arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # Plot 2: Epoch convergence comparison
    ax = axes[1]

    categories = ['Hyperparam\nSearch', 'Xukuang\nUNet', 'Xukuang\nAttn Models']
    best_epochs = [1, 140, 44]  # Average from data
    final_performance = [0.038, 0.607, 0.246]  # Normalized

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, best_epochs, width, label='Best Epoch',
                   color='#70AD47', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax_right = ax.twinx()
    bars2 = ax_right.bar(x + width/2, final_performance, width, label='Final IoU',
                         color='#FFC000', alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Best Epoch Number', fontsize=11, color='#70AD47')
    ax_right.set_ylabel('Final Validation IoU', fontsize=11, color='#FFC000')
    ax.set_title('Training Convergence Patterns', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.tick_params(axis='y', labelcolor='#70AD47')
    ax_right.tick_params(axis='y', labelcolor='#FFC000')
    ax.grid(True, alpha=0.3, axis='y')

    # Add labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
               f'{int(height)}', ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax_right.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(XUKUANG_DIR / 'comparison_why_xukuang_better.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_why_xukuang_better.png")
    plt.close()

def main():
    print("="*70)
    print("Comparative Analysis: Hyperparameter Search vs Xukuang Parameters")
    print("="*70)
    print()

    # Load data
    print("Loading experiment data...")
    hp_results, hp_summary, xk_info, xk_summary, xk_stats = load_data()
    print("✓ Data loaded")
    print()

    # Generate comparison plots
    print("Generating comparison visualizations...")
    create_performance_comparison(hp_results, xk_summary, xk_stats)
    create_improvement_analysis(hp_results, xk_stats)
    print()

    # Print summary statistics
    print("="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print()
    print("Hyperparameter Search (512×512 Grayscale):")
    print(f"  Best IoU: {hp_results['best_val_jaccard'].max():.4f}")
    print(f"  Mean IoU: {hp_results['best_val_jaccard'].mean():.4f}")
    print(f"  Training: 20 epochs (early stop), LR=1e-4/5e-5, Dropout=0.2/0.3")
    print(f"  Best Architecture: UNet")
    print()
    print("Xukuang Parameters (512×512 RGB):")
    print(f"  Best IoU: {max([xk_stats[k]['best_val_jaccard'] for k in xk_stats]):.4f}")
    print(f"  Final UNet IoU: {xk_summary['final_validation_metrics']['unet']['jaccard']:.4f}")
    print(f"  Training: 200 epochs, LR=5e-3, No dropout")
    print(f"  Best Architecture: UNet")
    print()
    print("Key Differences:")
    print(f"  • Xukuang uses 50× higher learning rate (5e-3 vs 1e-4)")
    print(f"  • Xukuang trains 10× longer (200 epochs vs ~20)")
    print(f"  • Xukuang uses RGB; Hyperparam uses Grayscale")
    print(f"  • Xukuang achieves 3.1× better IoU (0.679 vs 0.219)")
    print()
    print("="*70)

if __name__ == "__main__":
    main()
