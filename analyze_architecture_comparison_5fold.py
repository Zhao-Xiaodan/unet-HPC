#!/usr/bin/env python3
"""
Architecture Comparison Analysis and Visualization
===================================================

Analyzes and visualizes results from validate_architecture_comparison.py

Generates:
1. Performance comparison plots (Jaccard, Dice)
2. Training time analysis
3. Convergence comparison (learning curves)
4. Parameter efficiency analysis
5. Statistical significance testing
6. Comprehensive comparison report
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_comparison_results(results_dir):
    """Load comparison results from directory."""
    results_dir = Path(results_dir)

    # Load summary
    summary_file = results_dir / 'architecture_comparison_summary.json'

    if not summary_file.exists():
        print(f"❌ Summary file not found: {summary_file}")
        return None

    with open(summary_file, 'r') as f:
        summary = json.load(f)

    print(f"✅ Loaded results from: {results_dir}")
    print(f"   Architectures: {', '.join(summary['architectures_tested'])}")
    print(f"   Folds: {summary['comparison'][summary['architectures_tested'][0]]['n_folds']}")

    return summary, results_dir


def load_training_histories(results_dir, architectures):
    """Load training history CSVs for all architectures and folds."""
    results_dir = Path(results_dir)

    histories = {}

    for arch in architectures:
        arch_dir = results_dir / arch
        if not arch_dir.exists():
            continue

        histories[arch] = []

        # Find all fold directories
        fold_dirs = sorted(arch_dir.glob('fold_*'))

        for fold_dir in fold_dirs:
            history_file = fold_dir / 'history.csv'
            if history_file.exists():
                df = pd.read_csv(history_file)
                histories[arch].append(df)

    return histories


def create_performance_comparison(summary, save_path):
    """Create performance comparison visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Architecture Performance Comparison', fontsize=16, fontweight='bold')

    comparison = summary['comparison']
    architectures = summary['architectures_tested']

    # Architecture display names
    arch_labels = {
        'unet': 'U-Net',
        'resunet': 'ResUNet',
        'attention_resunet': 'Attention ResUNet'
    }

    # Colors
    colors = {
        'unet': '#1f77b4',
        'resunet': '#ff7f0e',
        'attention_resunet': '#2ca02c'
    }

    # 1. Best Val Jaccard by Fold (grouped bar chart)
    ax = axes[0, 0]

    x = np.arange(comparison[architectures[0]]['n_folds'])
    width = 0.25

    for i, arch in enumerate(architectures):
        if arch not in comparison:
            continue

        values = comparison[arch]['best_val_jacard']['values']
        offset = (i - 1) * width

        ax.bar(x + offset, values, width, label=arch_labels[arch],
               color=colors[arch], alpha=0.8)

    ax.set_xlabel('Fold', fontweight='bold')
    ax.set_ylabel('Best Val Jaccard', fontweight='bold')
    ax.set_title('Best Validation Jaccard by Fold', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(len(x))])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Box plot of performance distribution
    ax = axes[0, 1]

    data_for_box = []
    labels_for_box = []

    for arch in architectures:
        if arch not in comparison:
            continue

        data_for_box.append(comparison[arch]['best_val_jacard']['values'])
        labels_for_box.append(arch_labels[arch])

    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)

    for patch, arch in zip(bp['boxes'], architectures):
        patch.set_facecolor(colors[arch])
        patch.set_alpha(0.8)

    ax.set_ylabel('Best Val Jaccard', fontweight='bold')
    ax.set_title('Performance Distribution Across Folds', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add mean markers
    for i, arch in enumerate(architectures):
        if arch not in comparison:
            continue
        mean_val = comparison[arch]['best_val_jacard']['mean']
        ax.plot(i+1, mean_val, 'D', color='red', markersize=8, label='Mean' if i == 0 else '')

    ax.legend()

    # 3. Performance vs Parameters
    ax = axes[1, 0]

    for arch in architectures:
        if arch not in comparison:
            continue

        params = comparison[arch]['total_parameters'] / 1e6  # In millions
        mean_jac = comparison[arch]['best_val_jacard']['mean']
        std_jac = comparison[arch]['best_val_jacard']['std']

        ax.errorbar(params, mean_jac, yerr=std_jac, fmt='o', markersize=12,
                   label=arch_labels[arch], color=colors[arch], capsize=5,
                   linewidth=2)

        # Add text label
        ax.text(params, mean_jac + 0.02, arch_labels[arch],
               ha='center', fontsize=9)

    ax.set_xlabel('Model Parameters (Millions)', fontweight='bold')
    ax.set_ylabel('Best Val Jaccard (Mean ± Std)', fontweight='bold')
    ax.set_title('Performance vs Model Complexity', fontweight='bold')
    ax.grid(alpha=0.3)

    # 4. Training Time Comparison
    ax = axes[1, 1]

    arch_names = []
    epoch_times = []
    epoch_stds = []

    for arch in architectures:
        if arch not in comparison:
            continue

        arch_names.append(arch_labels[arch])
        epoch_times.append(comparison[arch]['avg_epoch_time_sec']['mean'])
        epoch_stds.append(comparison[arch]['avg_epoch_time_sec']['std'])

    x_pos = np.arange(len(arch_names))
    bars = ax.bar(x_pos, epoch_times, yerr=epoch_stds, capsize=5,
                  color=[colors[a] for a in architectures], alpha=0.8)

    ax.set_xlabel('Architecture', fontweight='bold')
    ax.set_ylabel('Average Epoch Time (seconds)', fontweight='bold')
    ax.set_title('Training Speed Comparison', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(arch_names)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, time in zip(bars, epoch_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{time:.1f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved performance comparison: {save_path}")

    plt.close()


def create_training_curves(histories, save_path):
    """Create training curves comparison."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Curves Comparison Across Architectures',
                 fontsize=16, fontweight='bold')

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

    for idx, arch in enumerate(histories.keys()):
        ax = axes[idx]

        # Plot all folds for this architecture
        for fold_idx, df in enumerate(histories[arch]):
            epochs = range(len(df))

            # Plot training and validation
            ax.plot(epochs, df['jacard_coef'], color=colors[arch],
                   alpha=0.3, linewidth=1)
            ax.plot(epochs, df['val_jacard_coef'], color=colors[arch],
                   alpha=0.6, linewidth=2, label=f'Fold {fold_idx+1}')

            # Mark best epoch
            best_epoch = df['val_jacard_coef'].idxmax()
            best_val = df['val_jacard_coef'].max()
            ax.plot(best_epoch, best_val, '*', color='gold',
                   markersize=12, markeredgecolor='black', markeredgewidth=0.5)

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Jaccard Coefficient', fontweight='bold')
        ax.set_title(f'{arch_labels[arch]}', fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

        # Add text explanation
        ax.text(0.02, 0.98, 'Light: Training\nDark: Validation\n⭐: Best',
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved training curves: {save_path}")

    plt.close()


def create_convergence_comparison(histories, save_path):
    """Create convergence speed comparison."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')

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

    # 1. Average validation Jaccard over epochs
    ax = axes[0, 0]

    for arch, fold_histories in histories.items():
        # Get max length
        max_epochs = max(len(df) for df in fold_histories)

        # Pad and average
        padded = []
        for df in fold_histories:
            vals = df['val_jacard_coef'].values
            if len(vals) < max_epochs:
                # Pad with last value
                vals = np.pad(vals, (0, max_epochs - len(vals)),
                            mode='edge')
            padded.append(vals)

        mean_curve = np.mean(padded, axis=0)
        std_curve = np.std(padded, axis=0)

        epochs = range(max_epochs)
        ax.plot(epochs, mean_curve, label=arch_labels[arch],
               color=colors[arch], linewidth=2)
        ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve,
                       color=colors[arch], alpha=0.2)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Val Jaccard (Mean ± Std)', fontweight='bold')
    ax.set_title('Average Convergence Curves', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Epochs to reach 90% of best performance
    ax = axes[0, 1]

    epochs_to_90 = {arch: [] for arch in histories.keys()}

    for arch, fold_histories in histories.items():
        for df in fold_histories:
            best_val = df['val_jacard_coef'].max()
            target = 0.9 * best_val

            # Find first epoch reaching target
            reached = df['val_jacard_coef'] >= target
            if reached.any():
                first_epoch = reached.idxmax()
                epochs_to_90[arch].append(first_epoch)

    # Box plot
    data_for_box = [epochs_to_90[arch] for arch in histories.keys()]
    labels_for_box = [arch_labels[arch] for arch in histories.keys()]

    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)

    for patch, arch in zip(bp['boxes'], histories.keys()):
        patch.set_facecolor(colors[arch])
        patch.set_alpha(0.8)

    ax.set_ylabel('Epochs to 90% Best Performance', fontweight='bold')
    ax.set_title('Convergence Speed', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 3. Overfitting progression
    ax = axes[1, 0]

    for arch, fold_histories in histories.items():
        # Calculate train/val gap over epochs
        gaps = []

        for df in fold_histories:
            train_vals = df['jacard_coef'].values
            val_vals = df['val_jacard_coef'].values
            gap = train_vals / np.maximum(val_vals, 1e-6)
            gaps.append(gap)

        # Pad and average
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
               color=colors[arch], linewidth=2)
        ax.fill_between(epochs, mean_gap - std_gap, mean_gap + std_gap,
                       color=colors[arch], alpha=0.2)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Overfitting Gap (Train/Val Jaccard)', fontweight='bold')
    ax.set_title('Overfitting Progression', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, label='No gap')

    # 4. Best epoch distribution
    ax = axes[1, 1]

    best_epochs = {arch: [] for arch in histories.keys()}

    for arch, fold_histories in histories.items():
        for df in fold_histories:
            best_epoch = df['val_jacard_coef'].idxmax()
            best_epochs[arch].append(best_epoch)

    # Histogram
    for arch in histories.keys():
        ax.hist(best_epochs[arch], bins=range(0, 21), alpha=0.5,
               label=arch_labels[arch], color=colors[arch], edgecolor='black')

    ax.set_xlabel('Best Epoch', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Best Epoch Distribution', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved convergence comparison: {save_path}")

    plt.close()


def statistical_significance_test(summary):
    """Perform statistical significance tests."""

    print("\n" + "="*80)
    print("📊 STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)

    comparison = summary['comparison']
    architectures = summary['architectures_tested']

    # Get Jaccard values
    data = {}
    for arch in architectures:
        if arch in comparison:
            data[arch] = comparison[arch]['best_val_jacard']['values']

    arch_labels = {
        'unet': 'U-Net',
        'resunet': 'ResUNet',
        'attention_resunet': 'Attention ResUNet'
    }

    # Pairwise t-tests
    print("\n🔬 Pairwise T-Tests (Best Val Jaccard):")
    print("-" * 80)

    results = []

    for i, arch1 in enumerate(architectures):
        for arch2 in architectures[i+1:]:
            if arch1 not in data or arch2 not in data:
                continue

            t_stat, p_value = stats.ttest_rel(data[arch1], data[arch2])

            mean1 = np.mean(data[arch1])
            mean2 = np.mean(data[arch2])
            diff = mean2 - mean1
            diff_pct = (diff / mean1) * 100

            significance = ''
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'

            print(f"\n{arch_labels[arch1]} vs {arch_labels[arch2]}:")
            print(f"  Mean difference: {diff:+.4f} ({diff_pct:+.1f}%)")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f} {significance}")

            if p_value < 0.05:
                winner = arch_labels[arch2] if diff > 0 else arch_labels[arch1]
                print(f"  ✅ SIGNIFICANT: {winner} performs better (p < 0.05)")
            else:
                print(f"  ❌ NOT SIGNIFICANT: No statistical difference (p ≥ 0.05)")

            results.append({
                'comparison': f'{arch_labels[arch1]} vs {arch_labels[arch2]}',
                'mean_diff': float(diff),
                'diff_pct': float(diff_pct),
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            })

    print("\n" + "="*80)

    return results


def generate_report(summary, results_dir, stat_results):
    """Generate comprehensive comparison report."""

    report_path = results_dir / 'ARCHITECTURE_COMPARISON_REPORT.md'

    comparison = summary['comparison']
    architectures = summary['architectures_tested']

    arch_labels = {
        'unet': 'U-Net',
        'resunet': 'ResUNet',
        'attention_resunet': 'Attention ResUNet'
    }

    with open(report_path, 'w') as f:
        f.write("# Architecture Comparison Report\n\n")
        f.write("## Executive Summary\n\n")

        # Find best architecture
        best_arch = max(architectures,
                       key=lambda a: comparison[a]['best_val_jacard']['mean'])
        best_jac = comparison[best_arch]['best_val_jacard']['mean']
        best_std = comparison[best_arch]['best_val_jacard']['std']

        baseline_jac = comparison['unet']['best_val_jacard']['mean']
        improvement = ((best_jac - baseline_jac) / baseline_jac) * 100

        f.write(f"**Best Performing Architecture:** {arch_labels[best_arch]}\n\n")
        f.write(f"- **Performance:** {best_jac:.4f} ± {best_std:.4f} (Jaccard)\n")
        f.write(f"- **Improvement over U-Net:** {improvement:+.1f}%\n")
        f.write(f"- **Statistical Significance:** See detailed analysis below\n\n")

        f.write("---\n\n")

        f.write("## Detailed Results\n\n")

        f.write("### Performance Comparison\n\n")
        f.write("| Architecture | Best Val Jaccard | Overfitting Gap | Avg Epoch Time | Parameters |\n")
        f.write("|--------------|------------------|-----------------|----------------|------------|\n")

        for arch in architectures:
            if arch not in comparison:
                continue

            stats = comparison[arch]
            jac = stats['best_val_jacard']['mean']
            jac_std = stats['best_val_jacard']['std']
            gap = stats['overfitting_gap']['mean']
            gap_std = stats['overfitting_gap']['std']
            time = stats['avg_epoch_time_sec']['mean']
            params = stats['total_parameters'] / 1e6

            f.write(f"| {arch_labels[arch]:<12} | ")
            f.write(f"{jac:.4f} ± {jac_std:.4f} | ")
            f.write(f"{gap:.2f}× ± {gap_std:.2f}× | ")
            f.write(f"{time:.1f}s | ")
            f.write(f"{params:.1f}M |\n")

        f.write("\n### Fold-by-Fold Results\n\n")

        for arch in architectures:
            if arch not in comparison:
                continue

            f.write(f"#### {arch_labels[arch]}\n\n")

            values = comparison[arch]['best_val_jacard']['values']
            best_epochs = comparison[arch]['best_epoch']['values']

            f.write("| Fold | Best Val Jaccard | Best Epoch |\n")
            f.write("|------|------------------|------------|\n")

            for i, (val, epoch) in enumerate(zip(values, best_epochs), 1):
                f.write(f"| {i} | {val:.4f} | {int(epoch)+1} |\n")

            f.write("\n")

        f.write("---\n\n")

        f.write("## Statistical Analysis\n\n")

        f.write("### Pairwise Comparisons\n\n")

        for result in stat_results:
            f.write(f"**{result['comparison']}**\n\n")
            f.write(f"- Mean difference: {result['mean_diff']:+.4f} ({result['diff_pct']:+.1f}%)\n")
            f.write(f"- p-value: {result['p_value']:.4f}\n")

            if result['significant']:
                f.write(f"- ✅ **SIGNIFICANT** (p < 0.05)\n")
            else:
                f.write(f"- ❌ Not significant (p ≥ 0.05)\n")

            f.write("\n")

        f.write("---\n\n")

        f.write("## Visualizations\n\n")
        f.write("![Performance Comparison](architecture_performance_comparison.png)\n\n")
        f.write("**Figure 1:** Performance comparison across architectures showing best validation Jaccard by fold, distribution, parameter efficiency, and training time.\n\n")
        f.write("![Training Curves](architecture_training_curves.png)\n\n")
        f.write("**Figure 2:** Training curves for each architecture across all folds, with gold stars marking best validation epochs.\n\n")
        f.write("![Convergence Analysis](architecture_convergence_analysis.png)\n\n")
        f.write("**Figure 3:** Convergence analysis including average curves, speed to 90% performance, overfitting progression, and best epoch distribution.\n\n")

        f.write("---\n\n")

        f.write("## Conclusions\n\n")

        f.write(f"1. **Best Architecture:** {arch_labels[best_arch]} achieved the highest performance ")
        f.write(f"({best_jac:.4f} ± {best_std:.4f} Jaccard)\n\n")

        # Check if improvement is significant
        sig_test = [r for r in stat_results if 'vs U-Net' in r['comparison'] and best_arch in r['comparison'].lower()]

        if sig_test and sig_test[0]['significant']:
            f.write(f"2. **Statistical Significance:** The improvement over U-Net is statistically significant (p < 0.05)\n\n")
        else:
            f.write(f"2. **Statistical Significance:** The improvement over U-Net is NOT statistically significant (p ≥ 0.05)\n\n")

        # Training time trade-off
        best_time = comparison[best_arch]['avg_epoch_time_sec']['mean']
        unet_time = comparison['unet']['avg_epoch_time_sec']['mean']
        time_overhead = ((best_time - unet_time) / unet_time) * 100

        f.write(f"3. **Computational Cost:** {arch_labels[best_arch]} has {time_overhead:+.1f}% time overhead per epoch\n\n")

        f.write("## Recommendations\n\n")

        if improvement > 5 and (not sig_test or sig_test[0]['significant']):
            f.write(f"✅ **RECOMMEND using {arch_labels[best_arch]}** for production:\n")
            f.write(f"- Substantial performance improvement ({improvement:+.1f}%)\n")
            f.write(f"- Acceptable computational overhead\n")
        elif improvement > 2:
            f.write(f"⚠️  **CONSIDER using {arch_labels[best_arch]}**:\n")
            f.write(f"- Moderate performance improvement ({improvement:+.1f}%)\n")
            f.write(f"- Evaluate if worth the computational cost\n")
        else:
            f.write(f"📊 **Continue using U-Net**:\n")
            f.write(f"- Minimal improvement from advanced architectures\n")
            f.write(f"- U-Net is simpler and faster\n")

        f.write("\n---\n\n")
        f.write("*Generated by analyze_architecture_comparison.py*\n")

    print(f"✅ Saved report: {report_path}")


def main():
    """Main analysis workflow."""

    if len(sys.argv) < 2:
        print("Usage: python analyze_architecture_comparison.py <results_directory>")
        print("\nExample: python analyze_architecture_comparison.py validation_arch_comparison_20251013_120000")
        return 1

    results_dir = sys.argv[1]

    # Load results
    result = load_comparison_results(results_dir)
    if result is None:
        return 1

    summary, results_dir = result

    # Load training histories
    histories = load_training_histories(results_dir, summary['architectures_tested'])

    # Create visualizations
    print("\n📊 Generating visualizations...")

    create_performance_comparison(
        summary,
        Path(results_dir) / 'architecture_performance_comparison.png'
    )

    if histories:
        create_training_curves(
            histories,
            Path(results_dir) / 'architecture_training_curves.png'
        )

        create_convergence_comparison(
            histories,
            Path(results_dir) / 'architecture_convergence_analysis.png'
        )

    # Statistical testing
    stat_results = statistical_significance_test(summary)

    # Generate report
    print("\n📝 Generating report...")
    generate_report(summary, Path(results_dir), stat_results)

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nGenerated files in {results_dir}:")
    print("  - ARCHITECTURE_COMPARISON_REPORT.md")
    print("  - architecture_performance_comparison.png")
    print("  - architecture_training_curves.png")
    print("  - architecture_convergence_analysis.png")

    return 0


if __name__ == "__main__":
    exit(main())
