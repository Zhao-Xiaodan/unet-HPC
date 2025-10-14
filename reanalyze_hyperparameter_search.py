#!/usr/bin/env python3
"""
Re-analyze Hyperparameter Search with Complete Dataset
=======================================================

Previous analysis was based on 12 configurations.
Now all 19/20 configurations are complete (except attention_resunet_lr1e-05_drop0.3_bs8).

This script:
1. Collects all results from fold directories
2. Regenerates all visualizations
3. Updates summary JSON
4. Regenerates comprehensive report

Author: Claude Code
Date: October 14, 2025
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configuration
HYPERPARAM_DIR = Path('./hyperparameter_search_20251013_154754')
OUTPUT_PREFIX = HYPERPARAM_DIR

# Baselines (from validation_arch_comparison)
BASELINES = {
    'unet': 0.6994,
    'resunet_baseline': 0.3995,
    'attention_resunet_baseline': 0.4176
}

# Color scheme
COLORS = {
    'learning_rate': {'1e-05': '#440154', '2e-05': '#31688e', '5e-05': '#35b779'},
    'dropout': {'0.3': '#440154', '0.4': '#31688e', '0.5': '#35b779'},
    'batch_size': {'4': '#440154', '8': '#fde724'}
}

def collect_all_results():
    """Collect results from all configuration directories."""
    all_results = []

    config_dirs = sorted([d for d in HYPERPARAM_DIR.iterdir()
                         if d.is_dir() and '_lr' in d.name])

    print(f"Found {len(config_dirs)} configuration directories")
    print()

    for config_dir in config_dirs:
        config_name = config_dir.name

        # Parse configuration from directory name
        # Format: {arch}_lr{lr}_drop{drop}_bs{bs}
        parts = config_name.split('_')

        # Extract architecture
        if config_name.startswith('attention_resunet'):
            arch = 'attention_resunet'
            idx = 2
        elif config_name.startswith('resunet'):
            arch = 'resunet'
            idx = 1
        else:
            arch = 'unet'
            idx = 1

        # Extract hyperparameters
        lr_str = parts[idx].replace('lr', '')
        drop_str = parts[idx+1].replace('drop', '')
        bs_str = parts[idx+2].replace('bs', '')

        lr = float(lr_str.replace('e-', 'e-0'))  # 1e-05, 2e-05, 5e-05
        dropout = float(drop_str)
        batch_size = int(bs_str)

        # Collect fold results
        fold_dirs = sorted([d for d in config_dir.iterdir()
                           if d.is_dir() and d.name.startswith('fold_')])

        config_fold_results = []

        for fold_dir in fold_dirs:
            results_file = fold_dir / 'results.json'

            if not results_file.exists():
                print(f"  ⚠ Missing results: {config_name}/{fold_dir.name}")
                continue

            with open(results_file, 'r') as f:
                fold_result = json.load(f)

            # Add configuration info
            fold_result['architecture'] = arch
            fold_result['config_name'] = config_name
            fold_result['config'] = {
                'learning_rate': lr,
                'dropout': dropout,
                'batch_size': batch_size
            }

            config_fold_results.append(fold_result)

        if config_fold_results:
            print(f"  ✓ {config_name}: {len(config_fold_results)} folds")
            all_results.extend(config_fold_results)
        else:
            print(f"  ✗ {config_name}: no results")

    print()
    print(f"Total results collected: {len(all_results)}")
    return all_results

def aggregate_config_results(fold_results):
    """Aggregate results by configuration."""
    config_results = {}

    for result in fold_results:
        config_name = result['config_name']

        if config_name not in config_results:
            config_results[config_name] = {
                'architecture': result['architecture'],
                'config_name': config_name,
                'config': result['config'],
                'fold_results': []
            }

        config_results[config_name]['fold_results'].append(result)

    # Calculate summary statistics for each configuration
    aggregated = []

    for config_name, config_data in config_results.items():
        fold_results = config_data['fold_results']

        best_jacards = [f['best_val_jacard'] for f in fold_results]
        best_epochs = [f['best_epoch'] for f in fold_results]
        overfitting_gaps = [f['overfitting_gap'] for f in fold_results]

        config_summary = {
            'architecture': config_data['architecture'],
            'config_name': config_name,
            **config_data['config'],
            'n_folds': len(fold_results),
            'mean_best_jacard': np.mean(best_jacards),
            'std_best_jacard': np.std(best_jacards),
            'min_best_jacard': np.min(best_jacards),
            'max_best_jacard': np.max(best_jacards),
            'mean_best_epoch': np.mean(best_epochs),
            'mean_overfitting_gap': np.mean(overfitting_gaps),
            'fold_results': fold_results
        }

        aggregated.append(config_summary)

    # Sort by mean Jaccard (descending)
    aggregated.sort(key=lambda x: x['mean_best_jacard'], reverse=True)

    return aggregated

def analyze_hyperparameter_effects(config_results):
    """Analyze effect of each hyperparameter."""
    df = pd.DataFrame([{
        'config_name': c['config_name'],
        'architecture': c['architecture'],
        'learning_rate': f"{c['learning_rate']:.0e}",
        'dropout': c['dropout'],
        'batch_size': c['batch_size'],
        'mean_jacard': c['mean_best_jacard']
    } for c in config_results])

    effects = {}

    # Learning rate effect
    lr_grouped = df.groupby('learning_rate')['mean_jacard'].agg(['mean', 'std', 'count'])
    effects['learning_rate'] = {
        'mean': lr_grouped['mean'].to_dict(),
        'std': lr_grouped['std'].to_dict(),
        'count': lr_grouped['count'].to_dict()
    }

    # Dropout effect
    drop_grouped = df.groupby('dropout')['mean_jacard'].agg(['mean', 'std', 'count'])
    effects['dropout'] = {
        'mean': drop_grouped['mean'].to_dict(),
        'std': drop_grouped['std'].to_dict(),
        'count': drop_grouped['count'].to_dict()
    }

    # Batch size effect
    bs_grouped = df.groupby('batch_size')['mean_jacard'].agg(['mean', 'std', 'count'])
    effects['batch_size'] = {
        'mean': bs_grouped['mean'].to_dict(),
        'std': bs_grouped['std'].to_dict(),
        'count': bs_grouped['count'].to_dict()
    }

    return effects

def plot_baseline_comparison(config_results, output_path):
    """Create baseline comparison plot."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort configurations by performance
    sorted_configs = sorted(config_results, key=lambda x: x['mean_best_jacard'], reverse=True)

    # Prepare data
    config_names = [c['config_name'].replace('resunet_', 'rs_').replace('attention_resunet', 'att_rs')
                   for c in sorted_configs]
    mean_jacards = [c['mean_best_jacard'] for c in sorted_configs]
    std_jacards = [c['std_best_jacard'] for c in sorted_configs]

    # Create bar plot
    x_pos = np.arange(len(config_names))
    bars = ax.bar(x_pos, mean_jacards, yerr=std_jacards,
                   capsize=5, alpha=0.7, color='#31688e', edgecolor='black')

    # Add baseline lines
    ax.axhline(y=BASELINES['resunet_baseline'], color='#fde724',
               linestyle='--', linewidth=2, label=f'ResUNet Baseline ({BASELINES["resunet_baseline"]:.4f})')
    ax.axhline(y=BASELINES['unet'], color='#35b779',
               linestyle='--', linewidth=2, label=f'U-Net Baseline ({BASELINES["unet"]:.4f})')

    if 'attention_resunet' in [c['architecture'] for c in sorted_configs]:
        ax.axhline(y=BASELINES['attention_resunet_baseline'], color='#440154',
                   linestyle='--', linewidth=2,
                   label=f'Attention ResUNet Baseline ({BASELINES["attention_resunet_baseline"]:.4f})')

    # Highlight best configuration
    best_idx = 0
    bars[best_idx].set_color('#35b779')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    # Labels
    ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Jaccard Coefficient', fontsize=14, fontweight='bold')
    ax.set_title('Hyperparameter Search Results - All Configurations',
                 fontsize=16, fontweight='bold', pad=20)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)

    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add value labels on top of bars
    for i, (bar, mean, std) in enumerate(zip(bars, mean_jacards, std_jacards)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=8, rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")

def plot_hyperparam_effects(config_results, output_path):
    """Create hyperparameter effects analysis plot."""
    df = pd.DataFrame([{
        'config_name': c['config_name'],
        'learning_rate': f"{c['learning_rate']:.0e}",
        'dropout': str(c['dropout']),
        'batch_size': str(c['batch_size']),
        'mean_jacard': c['mean_best_jacard'],
        'std_jacard': c['std_best_jacard']
    } for c in config_results])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Learning rate effect
    lr_data = df.groupby('learning_rate').agg({
        'mean_jacard': ['mean', 'std'],
        'config_name': 'count'
    }).reset_index()
    lr_data.columns = ['learning_rate', 'mean', 'std', 'count']
    lr_order = ['1e-05', '2e-05', '5e-05']
    lr_data = lr_data.set_index('learning_rate').reindex(lr_order).reset_index()

    axes[0].bar(range(len(lr_data)), lr_data['mean'],
                yerr=lr_data['std'], capsize=10,
                color=[COLORS['learning_rate'][lr] for lr in lr_data['learning_rate']],
                edgecolor='black', alpha=0.8)
    axes[0].set_xticks(range(len(lr_data)))
    axes[0].set_xticklabels(lr_data['learning_rate'])
    axes[0].set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Mean Jaccard Coefficient', fontsize=12, fontweight='bold')
    axes[0].set_title('Effect of Learning Rate', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add sample counts
    for i, (mean, count) in enumerate(zip(lr_data['mean'], lr_data['count'])):
        axes[0].text(i, mean + lr_data['std'].iloc[i], f'n={int(count)}',
                    ha='center', va='bottom', fontsize=10)

    # Dropout effect
    drop_data = df.groupby('dropout').agg({
        'mean_jacard': ['mean', 'std'],
        'config_name': 'count'
    }).reset_index()
    drop_data.columns = ['dropout', 'mean', 'std', 'count']

    axes[1].bar(range(len(drop_data)), drop_data['mean'],
                yerr=drop_data['std'], capsize=10,
                color=[COLORS['dropout'][d] for d in drop_data['dropout']],
                edgecolor='black', alpha=0.8)
    axes[1].set_xticks(range(len(drop_data)))
    axes[1].set_xticklabels(drop_data['dropout'])
    axes[1].set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mean Jaccard Coefficient', fontsize=12, fontweight='bold')
    axes[1].set_title('Effect of Dropout', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    for i, (mean, count) in enumerate(zip(drop_data['mean'], drop_data['count'])):
        axes[1].text(i, mean + drop_data['std'].iloc[i], f'n={int(count)}',
                    ha='center', va='bottom', fontsize=10)

    # Batch size effect
    bs_data = df.groupby('batch_size').agg({
        'mean_jacard': ['mean', 'std'],
        'config_name': 'count'
    }).reset_index()
    bs_data.columns = ['batch_size', 'mean', 'std', 'count']

    axes[2].bar(range(len(bs_data)), bs_data['mean'],
                yerr=bs_data['std'], capsize=10,
                color=[COLORS['batch_size'][bs] for bs in bs_data['batch_size']],
                edgecolor='black', alpha=0.8)
    axes[2].set_xticks(range(len(bs_data)))
    axes[2].set_xticklabels(bs_data['batch_size'])
    axes[2].set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Mean Jaccard Coefficient', fontsize=12, fontweight='bold')
    axes[2].set_title('Effect of Batch Size', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    for i, (mean, count) in enumerate(zip(bs_data['mean'], bs_data['count'])):
        axes[2].text(i, mean + bs_data['std'].iloc[i], f'n={int(count)}',
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")

def plot_hyperparam_heatmaps(config_results, output_path):
    """Create heatmaps showing hyperparameter interactions."""
    # Prepare data
    df = pd.DataFrame([{
        'learning_rate': f"{c['learning_rate']:.0e}",
        'dropout': c['dropout'],
        'batch_size': c['batch_size'],
        'mean_jacard': c['mean_best_jacard']
    } for c in config_results])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # LR vs Dropout (averaged over batch size)
    pivot1 = df.groupby(['learning_rate', 'dropout'])['mean_jacard'].mean().unstack()
    pivot1 = pivot1.reindex(['1e-05', '2e-05', '5e-05'])

    sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=0.7, cbar_kws={'label': 'Mean Jaccard'},
                ax=axes[0], linewidths=1, linecolor='white')
    axes[0].set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[0].set_title('LR × Dropout\n(averaged over batch size)',
                     fontsize=13, fontweight='bold')

    # LR vs Batch Size (averaged over dropout)
    pivot2 = df.groupby(['learning_rate', 'batch_size'])['mean_jacard'].mean().unstack()
    pivot2 = pivot2.reindex(['1e-05', '2e-05', '5e-05'])

    sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=0.7, cbar_kws={'label': 'Mean Jaccard'},
                ax=axes[1], linewidths=1, linecolor='white')
    axes[1].set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('LR × Batch Size\n(averaged over dropout)',
                     fontsize=13, fontweight='bold')

    # Dropout vs Batch Size (averaged over LR)
    pivot3 = df.groupby(['dropout', 'batch_size'])['mean_jacard'].mean().unstack()

    sns.heatmap(pivot3, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=0.7, cbar_kws={'label': 'Mean Jaccard'},
                ax=axes[2], linewidths=1, linecolor='white')
    axes[2].set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Dropout Rate', fontsize=12, fontweight='bold')
    axes[2].set_title('Dropout × Batch Size\n(averaged over LR)',
                     fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")

def save_summary_json(config_results, hyperparam_effects, output_path):
    """Save summary JSON."""
    best_config = config_results[0]  # Already sorted by performance

    summary = {
        'n_configurations': len(config_results),
        'best_config': best_config,
        'all_configs': config_results,
        'hyperparameter_effects': hyperparam_effects,
        'baselines': BASELINES
    }

    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    summary = convert_types(summary)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  ✓ Saved: {output_path.name}")

def main():
    print("=" * 80)
    print("RE-ANALYZING HYPERPARAMETER SEARCH WITH COMPLETE DATASET")
    print("=" * 80)
    print()

    # Collect all results
    print("Step 1: Collecting results from all configurations...")
    fold_results = collect_all_results()
    print()

    # Aggregate by configuration
    print("Step 2: Aggregating results by configuration...")
    config_results = aggregate_config_results(fold_results)
    print(f"  ✓ Aggregated {len(config_results)} configurations")
    print()

    # Analyze hyperparameter effects
    print("Step 3: Analyzing hyperparameter effects...")
    hyperparam_effects = analyze_hyperparameter_effects(config_results)
    print("  ✓ Analysis complete")
    print()

    # Generate visualizations
    print("Step 4: Generating visualizations...")
    plot_baseline_comparison(config_results, OUTPUT_PREFIX / 'baseline_comparison.png')
    plot_hyperparam_effects(config_results, OUTPUT_PREFIX / 'hyperparam_effects_analysis.png')
    plot_hyperparam_heatmaps(config_results, OUTPUT_PREFIX / 'hyperparam_heatmaps.png')
    print()

    # Save summary JSON
    print("Step 5: Saving summary JSON...")
    save_summary_json(config_results, hyperparam_effects,
                     OUTPUT_PREFIX / 'hyperparameter_search_summary.json')
    print()

    # Print best configuration
    print("=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    best = config_results[0]
    print(f"Configuration: {best['config_name']}")
    print(f"Architecture: {best['architecture']}")
    print(f"Learning Rate: {best['learning_rate']:.0e}")
    print(f"Dropout: {best['dropout']}")
    print(f"Batch Size: {best['batch_size']}")
    print(f"Mean Jaccard: {best['mean_best_jacard']:.4f} ± {best['std_best_jacard']:.4f}")
    print(f"Range: [{best['min_best_jacard']:.4f}, {best['max_best_jacard']:.4f}]")
    print(f"Mean Best Epoch: {best['mean_best_epoch']:.1f}")
    print(f"Mean Overfitting Gap: {best['mean_overfitting_gap']:.2f}%")
    print()

    # Print hyperparameter effects summary
    print("=" * 80)
    print("HYPERPARAMETER EFFECTS SUMMARY")
    print("=" * 80)
    print()
    print("Learning Rate:")
    for lr, mean_j in sorted(hyperparam_effects['learning_rate']['mean'].items()):
        std_j = hyperparam_effects['learning_rate']['std'][lr]
        count = hyperparam_effects['learning_rate']['count'][lr]
        print(f"  {lr:>6}: {mean_j:.4f} ± {std_j:.4f} (n={count})")
    print()
    print("Dropout:")
    for drop, mean_j in sorted(hyperparam_effects['dropout']['mean'].items()):
        std_j = hyperparam_effects['dropout']['std'][drop]
        count = hyperparam_effects['dropout']['count'][drop]
        print(f"  {drop:.1f}: {mean_j:.4f} ± {std_j:.4f} (n={count})")
    print()
    print("Batch Size:")
    for bs, mean_j in sorted(hyperparam_effects['batch_size']['mean'].items()):
        std_j = hyperparam_effects['batch_size']['std'][bs]
        count = hyperparam_effects['batch_size']['count'][bs]
        print(f"  {bs:>2}: {mean_j:.4f} ± {std_j:.4f} (n={count})")
    print()

    print("=" * 80)
    print("RE-ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(f"Output directory: {OUTPUT_PREFIX}")
    print()
    print("Generated files:")
    print("  ✓ baseline_comparison.png (updated)")
    print("  ✓ hyperparam_effects_analysis.png (updated)")
    print("  ✓ hyperparam_heatmaps.png (updated)")
    print("  ✓ hyperparameter_search_summary.json (updated)")
    print()
    print("Next: Run generate report script to create updated REPORT.md")
    print()

if __name__ == '__main__':
    main()
