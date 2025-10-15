#!/usr/bin/env python3
"""
Analyze hyperparameter optimization results and generate summary reports.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import argparse

def load_results(summary_file):
    """Load and clean results data."""
    df = pd.read_csv(summary_file)

    # Clean data - remove failed experiments
    df = df[df['training_successful'] == True]

    # Convert data types
    df['learning_rate'] = df['learning_rate'].astype(float)
    df['batch_size'] = df['batch_size'].astype(int)
    df['best_val_jaccard'] = df['best_val_jaccard'].astype(float)
    df['val_loss_stability'] = df['val_loss_stability'].astype(float)

    return df

def create_performance_heatmaps(df, output_dir):
    """Create heatmaps showing performance across hyperparameters."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hyperparameter Optimization Results - Performance Heatmaps', fontsize=16, fontweight='bold')

    architectures = df['architecture'].unique()

    for i, arch in enumerate(architectures):
        arch_data = df[df['architecture'] == arch]

        # Create pivot table for jaccard performance
        pivot_jaccard = arch_data.pivot_table(
            values='best_val_jaccard',
            index='learning_rate',
            columns='batch_size',
            aggfunc='mean'
        )

        # Create pivot table for stability
        pivot_stability = arch_data.pivot_table(
            values='val_loss_stability',
            index='learning_rate',
            columns='batch_size',
            aggfunc='mean'
        )

        # Plot jaccard heatmap
        sns.heatmap(
            pivot_jaccard,
            annot=True,
            fmt='.4f',
            cmap='viridis',
            ax=axes[0, i]
        )
        axes[0, i].set_title(f'{arch} - Best Val Jaccard')
        axes[0, i].set_xlabel('Batch Size')
        axes[0, i].set_ylabel('Learning Rate')

        # Plot stability heatmap
        sns.heatmap(
            pivot_stability,
            annot=True,
            fmt='.4f',
            cmap='viridis_r',  # Reverse colormap (lower is better for stability)
            ax=axes[1, i]
        )
        axes[1, i].set_title(f'{arch} - Val Loss Stability (Lower=Better)')
        axes[1, i].set_xlabel('Batch Size')
        axes[1, i].set_ylabel('Learning Rate')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hyperparameter_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_plots(df, output_dir):
    """Create comparison plots across architectures."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Architecture Comparison Across Hyperparameters', fontsize=16, fontweight='bold')

    # Performance vs Learning Rate
    for arch in df['architecture'].unique():
        arch_data = df[df['architecture'] == arch]
        lr_performance = arch_data.groupby('learning_rate')['best_val_jaccard'].mean()
        axes[0, 0].plot(lr_performance.index, lr_performance.values, marker='o', label=arch)

    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Best Val Jaccard')
    axes[0, 0].set_title('Performance vs Learning Rate')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    # Performance vs Batch Size
    for arch in df['architecture'].unique():
        arch_data = df[df['architecture'] == arch]
        bs_performance = arch_data.groupby('batch_size')['best_val_jaccard'].mean()
        axes[0, 1].plot(bs_performance.index, bs_performance.values, marker='o', label=arch)

    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Best Val Jaccard')
    axes[0, 1].set_title('Performance vs Batch Size')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Stability vs Learning Rate
    for arch in df['architecture'].unique():
        arch_data = df[df['architecture'] == arch]
        lr_stability = arch_data.groupby('learning_rate')['val_loss_stability'].mean()
        axes[1, 0].plot(lr_stability.index, lr_stability.values, marker='o', label=arch)

    axes[1, 0].set_xlabel('Learning Rate')
    axes[1, 0].set_ylabel('Val Loss Stability (Lower=Better)')
    axes[1, 0].set_title('Stability vs Learning Rate')
    axes[1, 0].legend()
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Stability vs Batch Size
    for arch in df['architecture'].unique():
        arch_data = df[df['architecture'] == arch]
        bs_stability = arch_data.groupby('batch_size')['val_loss_stability'].mean()
        axes[1, 1].plot(bs_stability.index, bs_stability.values, marker='o', label=arch)

    axes[1, 1].set_xlabel('Batch Size')
    axes[1, 1].set_ylabel('Val Loss Stability (Lower=Better)')
    axes[1, 1].set_title('Stability vs Batch Size')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'architecture_comparisons.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(df, output_dir):
    """Generate text summary report."""

    report_lines = []
    report_lines.append("HYPERPARAMETER OPTIMIZATION SUMMARY REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"Total successful experiments: {len(df)}")
    report_lines.append(f"Architectures tested: {', '.join(df['architecture'].unique())}")
    report_lines.append(f"Learning rates tested: {', '.join([str(x) for x in sorted(df['learning_rate'].unique())])}")
    report_lines.append(f"Batch sizes tested: {', '.join([str(x) for x in sorted(df['batch_size'].unique())])}")
    report_lines.append("")

    # Best overall configuration
    best_idx = df['best_val_jaccard'].idxmax()
    best_config = df.loc[best_idx]

    report_lines.append("BEST OVERALL CONFIGURATION:")
    report_lines.append("-" * 30)
    report_lines.append(f"Architecture: {best_config['architecture']}")
    report_lines.append(f"Learning Rate: {best_config['learning_rate']}")
    report_lines.append(f"Batch Size: {best_config['batch_size']}")
    report_lines.append(f"Best Val Jaccard: {best_config['best_val_jaccard']:.4f}")
    report_lines.append(f"Stability (std dev): {best_config['val_loss_stability']:.4f}")
    report_lines.append("")

    # Best configuration per architecture
    report_lines.append("BEST CONFIGURATION PER ARCHITECTURE:")
    report_lines.append("-" * 40)

    for arch in df['architecture'].unique():
        arch_data = df[df['architecture'] == arch]
        best_arch_idx = arch_data['best_val_jaccard'].idxmax()
        best_arch_config = arch_data.loc[best_arch_idx]

        report_lines.append(f"\n{arch}:")
        report_lines.append(f"  Learning Rate: {best_arch_config['learning_rate']}")
        report_lines.append(f"  Batch Size: {best_arch_config['batch_size']}")
        report_lines.append(f"  Best Val Jaccard: {best_arch_config['best_val_jaccard']:.4f}")
        report_lines.append(f"  Stability: {best_arch_config['val_loss_stability']:.4f}")

    report_lines.append("")

    # Learning rate analysis
    report_lines.append("LEARNING RATE ANALYSIS:")
    report_lines.append("-" * 25)
    lr_analysis = df.groupby('learning_rate').agg({
        'best_val_jaccard': ['mean', 'std'],
        'val_loss_stability': ['mean', 'std']
    }).round(4)

    for lr in sorted(df['learning_rate'].unique()):
        lr_data = df[df['learning_rate'] == lr]
        report_lines.append(f"\nLearning Rate {lr}:")
        report_lines.append(f"  Avg Jaccard: {lr_data['best_val_jaccard'].mean():.4f} ± {lr_data['best_val_jaccard'].std():.4f}")
        report_lines.append(f"  Avg Stability: {lr_data['val_loss_stability'].mean():.4f} ± {lr_data['val_loss_stability'].std():.4f}")

    # Batch size analysis
    report_lines.append("")
    report_lines.append("BATCH SIZE ANALYSIS:")
    report_lines.append("-" * 22)

    for bs in sorted(df['batch_size'].unique()):
        bs_data = df[df['batch_size'] == bs]
        report_lines.append(f"\nBatch Size {bs}:")
        report_lines.append(f"  Avg Jaccard: {bs_data['best_val_jaccard'].mean():.4f} ± {bs_data['best_val_jaccard'].std():.4f}")
        report_lines.append(f"  Avg Stability: {bs_data['val_loss_stability'].mean():.4f} ± {bs_data['val_loss_stability'].std():.4f}")

    # Save report
    with open(os.path.join(output_dir, 'hyperparameter_summary_report.txt'), 'w') as f:
        f.write('\n'.join(report_lines))

    # Print to console
    print('\n'.join(report_lines))

def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter optimization results')
    parser.add_argument('--summary_file', required=True, help='Path to summary CSV file')
    parser.add_argument('--output_dir', required=True, help='Output directory for plots and reports')

    args = parser.parse_args()

    # Load results
    df = load_results(args.summary_file)

    if len(df) == 0:
        print("No successful experiments found!")
        return

    print(f"Analyzing {len(df)} successful experiments...")

    # Create visualizations
    create_performance_heatmaps(df, args.output_dir)
    create_comparison_plots(df, args.output_dir)

    # Generate summary report
    generate_summary_report(df, args.output_dir)

    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
