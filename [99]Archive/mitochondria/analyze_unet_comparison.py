#!/usr/bin/env python3
"""
Comprehensive analysis and visualization of U-Net architecture comparison
for mitochondria segmentation.

This script analyzes the training results from three U-Net variants:
- Standard U-Net
- Attention U-Net
- Attention Residual U-Net

Generates comparison plots and performance analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set up matplotlib for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_training_data(results_dir):
    """Load training history data from CSV files."""
    data = {}

    # File mappings
    files = {
        'UNet': 'unet_history_df.csv',
        'Attention_UNet': 'att_unet_history_df.csv',
        'Attention_ResUNet': 'custom_code_att_res_unet_history_df.csv'
    }

    for model_name, filename in files.items():
        filepath = Path(results_dir) / filename
        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0)
            data[model_name] = df
            print(f"Loaded {model_name}: {len(df)} epochs")
        else:
            print(f"Warning: {filepath} not found")

    return data

def create_comparison_plots(data, output_dir):
    """Create comprehensive comparison plots."""

    # Define colors for each model
    colors = {
        'UNet': '#1f77b4',
        'Attention_UNet': '#ff7f0e',
        'Attention_ResUNet': '#2ca02c'
    }

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('U-Net Architecture Comparison for Mitochondria Segmentation', fontsize=16, fontweight='bold')

    # 1. Training Loss Comparison
    ax = axes[0, 0]
    for model_name, df in data.items():
        epochs = range(1, len(df) + 1)
        ax.plot(epochs, df['loss'], label=f'{model_name}', color=colors[model_name], linewidth=2)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Binary Focal Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Validation Loss Comparison
    ax = axes[0, 1]
    for model_name, df in data.items():
        epochs = range(1, len(df) + 1)
        ax.plot(epochs, df['val_loss'], label=f'{model_name}', color=colors[model_name], linewidth=2)
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Binary Focal Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Training Jaccard Coefficient
    ax = axes[0, 2]
    for model_name, df in data.items():
        epochs = range(1, len(df) + 1)
        ax.plot(epochs, df['jacard_coef'], label=f'{model_name}', color=colors[model_name], linewidth=2)
    ax.set_title('Training Jaccard Coefficient')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Jaccard Coefficient (IoU)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Validation Jaccard Coefficient
    ax = axes[1, 0]
    for model_name, df in data.items():
        epochs = range(1, len(df) + 1)
        ax.plot(epochs, df['val_jacard_coef'], label=f'{model_name}', color=colors[model_name], linewidth=2)
    ax.set_title('Validation Jaccard Coefficient')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Jaccard Coefficient (IoU)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Training Accuracy
    ax = axes[1, 1]
    for model_name, df in data.items():
        epochs = range(1, len(df) + 1)
        ax.plot(epochs, df['accuracy'], label=f'{model_name}', color=colors[model_name], linewidth=2)
    ax.set_title('Training Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Validation Accuracy
    ax = axes[1, 2]
    for model_name, df in data.items():
        epochs = range(1, len(df) + 1)
        ax.plot(epochs, df['val_accuracy'], label=f'{model_name}', color=colors[model_name], linewidth=2)
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / 'unet_architecture_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")

    return fig

def create_performance_summary_plot(data, output_dir):
    """Create a performance summary bar chart."""

    # Calculate performance metrics
    metrics = {}
    for model_name, df in data.items():
        metrics[model_name] = {
            'best_val_jaccard': df['val_jacard_coef'].max(),
            'best_val_jaccard_epoch': df['val_jacard_coef'].idxmax() + 1,
            'final_val_loss': df['val_loss'].iloc[-1],
            'final_train_loss': df['loss'].iloc[-1],
            'final_val_accuracy': df['val_accuracy'].iloc[-1],
            'convergence_stability': df['val_loss'].iloc[-10:].std()  # Std dev of last 10 epochs
        }

    # Create performance summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Performance Summary', fontsize=14, fontweight='bold')

    models = list(metrics.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Best Validation Jaccard
    ax = axes[0]
    jaccard_scores = [metrics[model]['best_val_jaccard'] for model in models]
    bars = ax.bar(models, jaccard_scores, color=colors)
    ax.set_title('Best Validation Jaccard Coefficient')
    ax.set_ylabel('Jaccard Coefficient')
    ax.set_ylim(0, max(jaccard_scores) * 1.1)

    # Add value labels on bars
    for bar, score in zip(bars, jaccard_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

    # Final Validation Loss
    ax = axes[1]
    val_losses = [metrics[model]['final_val_loss'] for model in models]
    bars = ax.bar(models, val_losses, color=colors)
    ax.set_title('Final Validation Loss')
    ax.set_ylabel('Binary Focal Loss')
    ax.set_ylim(0, max(val_losses) * 1.1)

    # Add value labels on bars
    for bar, loss in zip(bars, val_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')

    # Convergence Stability (lower is better)
    ax = axes[2]
    stabilities = [metrics[model]['convergence_stability'] for model in models]
    bars = ax.bar(models, stabilities, color=colors)
    ax.set_title('Convergence Stability\n(Validation Loss Std Dev - Last 10 Epochs)')
    ax.set_ylabel('Standard Deviation')
    ax.set_ylim(0, max(stabilities) * 1.1)

    # Add value labels on bars
    for bar, stab in zip(bars, stabilities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stabilities) * 0.02,
                f'{stab:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / 'unet_performance_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved performance summary: {output_path}")

    return metrics

def analyze_training_dynamics(data, output_dir):
    """Analyze training dynamics and convergence patterns."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Dynamics Analysis', fontsize=14, fontweight='bold')

    colors = {
        'UNet': '#1f77b4',
        'Attention_UNet': '#ff7f0e',
        'Attention_ResUNet': '#2ca02c'
    }

    # 1. Loss convergence curves (smoothed)
    ax = axes[0, 0]
    for model_name, df in data.items():
        epochs = range(1, len(df) + 1)
        # Apply moving average for smoother curves
        window = 5
        train_loss_smooth = df['loss'].rolling(window=window, center=True).mean()
        val_loss_smooth = df['val_loss'].rolling(window=window, center=True).mean()

        ax.plot(epochs, train_loss_smooth, label=f'{model_name} (Train)',
                color=colors[model_name], linestyle='-', alpha=0.8)
        ax.plot(epochs, val_loss_smooth, label=f'{model_name} (Val)',
                color=colors[model_name], linestyle='--', alpha=0.8)

    ax.set_title('Smoothed Loss Convergence (5-epoch moving average)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Binary Focal Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Overfitting analysis (Val loss - Train loss)
    ax = axes[0, 1]
    for model_name, df in data.items():
        epochs = range(1, len(df) + 1)
        overfitting_gap = df['val_loss'] - df['loss']
        ax.plot(epochs, overfitting_gap, label=f'{model_name}',
                color=colors[model_name], linewidth=2)

    ax.set_title('Overfitting Analysis\n(Validation Loss - Training Loss)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Difference')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Learning rate effectiveness (Loss improvement per epoch)
    ax = axes[1, 0]
    for model_name, df in data.items():
        epochs = range(2, len(df) + 1)
        loss_improvement = -df['val_loss'].diff().iloc[1:]  # Negative diff means improvement
        ax.plot(epochs, loss_improvement, label=f'{model_name}',
                color=colors[model_name], alpha=0.7)

    ax.set_title('Learning Progress\n(Validation Loss Improvement per Epoch)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Improvement')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Performance plateau analysis
    ax = axes[1, 1]
    for model_name, df in data.items():
        epochs = range(1, len(df) + 1)
        # Calculate rolling standard deviation to detect plateaus
        window = 10
        val_jaccard_std = df['val_jacard_coef'].rolling(window=window).std()
        ax.plot(epochs, val_jaccard_std, label=f'{model_name}',
                color=colors[model_name], linewidth=2)

    ax.set_title('Performance Plateau Detection\n(10-epoch Rolling Std of Val Jaccard)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Standard Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / 'unet_training_dynamics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training dynamics analysis: {output_path}")

    return fig

def generate_performance_statistics(data):
    """Generate detailed performance statistics."""

    stats = {}

    for model_name, df in data.items():
        # Basic statistics
        best_val_jaccard = df['val_jacard_coef'].max()
        best_val_jaccard_epoch = df['val_jacard_coef'].idxmax() + 1
        final_val_jaccard = df['val_jacard_coef'].iloc[-1]

        # Training efficiency
        epochs_to_best = best_val_jaccard_epoch
        training_efficiency = best_val_jaccard / epochs_to_best

        # Stability metrics
        last_10_val_loss_std = df['val_loss'].iloc[-10:].std()
        last_10_val_jaccard_std = df['val_jacard_coef'].iloc[-10:].std()

        # Overfitting analysis
        avg_overfitting_gap = (df['val_loss'] - df['loss']).mean()
        final_overfitting_gap = df['val_loss'].iloc[-1] - df['loss'].iloc[-1]

        # Convergence analysis
        val_loss_improvement = df['val_loss'].iloc[0] - df['val_loss'].iloc[-1]
        train_loss_improvement = df['loss'].iloc[0] - df['loss'].iloc[-1]

        stats[model_name] = {
            'best_val_jaccard': best_val_jaccard,
            'best_val_jaccard_epoch': best_val_jaccard_epoch,
            'final_val_jaccard': final_val_jaccard,
            'final_val_loss': df['val_loss'].iloc[-1],
            'final_train_loss': df['loss'].iloc[-1],
            'training_efficiency': training_efficiency,
            'convergence_stability_loss': last_10_val_loss_std,
            'convergence_stability_jaccard': last_10_val_jaccard_std,
            'avg_overfitting_gap': avg_overfitting_gap,
            'final_overfitting_gap': final_overfitting_gap,
            'val_loss_improvement': val_loss_improvement,
            'train_loss_improvement': train_loss_improvement,
            'epochs_trained': len(df)
        }

    return stats

def main():
    """Main analysis function."""

    results_dir = 'mitochondria_segmentation_20250925_133928'
    output_dir = '.'

    print("=" * 60)
    print("U-Net Architecture Comparison Analysis")
    print("=" * 60)

    # Load data
    print("\n1. Loading training data...")
    data = load_training_data(results_dir)

    if not data:
        print("Error: No training data found!")
        return

    # Create comparison plots
    print("\n2. Creating comparison plots...")
    create_comparison_plots(data, output_dir)

    # Create performance summary
    print("\n3. Creating performance summary...")
    performance_metrics = create_performance_summary_plot(data, output_dir)

    # Analyze training dynamics
    print("\n4. Analyzing training dynamics...")
    analyze_training_dynamics(data, output_dir)

    # Generate detailed statistics
    print("\n5. Generating performance statistics...")
    stats = generate_performance_statistics(data)

    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    for model_name, model_stats in stats.items():
        print(f"\n{model_name}:")
        print(f"  Best Val Jaccard: {model_stats['best_val_jaccard']:.4f} (epoch {model_stats['best_val_jaccard_epoch']})")
        print(f"  Final Val Loss: {model_stats['final_val_loss']:.4f}")
        print(f"  Training Efficiency: {model_stats['training_efficiency']:.6f} (Jaccard/epoch)")
        print(f"  Convergence Stability: {model_stats['convergence_stability_loss']:.4f}")
        print(f"  Overfitting Gap: {model_stats['final_overfitting_gap']:.4f}")

    # Ranking
    print("\n" + "=" * 60)
    print("MODEL RANKING")
    print("=" * 60)

    # Rank by best validation Jaccard
    ranked_by_jaccard = sorted(stats.items(), key=lambda x: x[1]['best_val_jaccard'], reverse=True)
    print("\nBest Validation Jaccard:")
    for i, (model_name, model_stats) in enumerate(ranked_by_jaccard, 1):
        print(f"  {i}. {model_name}: {model_stats['best_val_jaccard']:.4f}")

    # Rank by stability (lower std is better)
    ranked_by_stability = sorted(stats.items(), key=lambda x: x[1]['convergence_stability_loss'])
    print("\nMost Stable Training:")
    for i, (model_name, model_stats) in enumerate(ranked_by_stability, 1):
        print(f"  {i}. {model_name}: {model_stats['convergence_stability_loss']:.4f} (std dev)")

    print(f"\n✓ Analysis complete! Plots saved to current directory.")

if __name__ == "__main__":
    main()