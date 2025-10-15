#!/usr/bin/env python3
"""
Analysis script for Xukuang parameters experiment
Compares UNet, Attention UNet, and Attention ResUNet on shrunk dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Define experiment directory
EXP_DIR = Path("xukuang_params_shrunk_20251015_071224")

def load_experiment_data():
    """Load all experiment data files"""

    # Load metadata
    with open(EXP_DIR / "EXPERIMENT_INFO.json", 'r') as f:
        exp_info = json.load(f)

    with open(EXP_DIR / "TRAINING_SUMMARY.json", 'r') as f:
        summary = json.load(f)

    # Load training histories
    unet_hist = pd.read_csv(EXP_DIR / "unet_history.csv")
    attn_unet_hist = pd.read_csv(EXP_DIR / "attention_unet_history.csv")
    attn_resunet_hist = pd.read_csv(EXP_DIR / "attention_resunet_history.csv")

    return exp_info, summary, {
        'UNet': unet_hist,
        'Attention UNet': attn_unet_hist,
        'Attention ResUNet': attn_resunet_hist
    }

def plot_training_curves(histories, summary):
    """Create comprehensive training curves comparison"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training History Comparison - Xukuang Parameters on Shrunk Dataset',
                 fontsize=16, fontweight='bold', y=0.995)

    metrics = [
        ('loss', 'Training Loss'),
        ('accuracy', 'Training Accuracy'),
        ('jacard_coef', 'Training Jaccard (IoU)')
    ]
    val_metrics = [
        ('val_loss', 'Validation Loss'),
        ('val_accuracy', 'Validation Accuracy'),
        ('val_jacard_coef', 'Validation Jaccard (IoU)')
    ]

    colors = {'UNet': '#2E86AB', 'Attention UNet': '#A23B72', 'Attention ResUNet': '#F18F01'}

    # Plot training metrics
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[0, idx]
        for name, hist in histories.items():
            ax.plot(hist[metric], label=name, color=colors[name], linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    # Plot validation metrics
    for idx, (metric, title) in enumerate(val_metrics):
        ax = axes[1, idx]
        for name, hist in histories.items():
            ax.plot(hist[metric], label=name, color=colors[name], linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(EXP_DIR / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: training_curves_comparison.png")
    plt.close()

def plot_final_metrics_comparison(summary):
    """Create bar chart comparing final validation metrics"""

    models = ['UNet', 'Attention UNet', 'Attention ResUNet']
    metrics_data = summary['final_validation_metrics']

    # Extract metrics
    losses = [metrics_data['unet']['loss'],
              metrics_data['attention_unet']['loss'],
              metrics_data['attention_resunet']['loss']]
    accuracies = [metrics_data['unet']['accuracy'],
                  metrics_data['attention_unet']['accuracy'],
                  metrics_data['attention_resunet']['accuracy']]
    jaccards = [metrics_data['unet']['jaccard'],
                metrics_data['attention_unet']['jaccard'],
                metrics_data['attention_resunet']['jaccard']]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Final Validation Metrics Comparison', fontsize=16, fontweight='bold')

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    # Loss comparison (lower is better)
    axes[0].bar(models, losses, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Validation Loss (Lower is Better)', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(losses):
        axes[0].text(i, v + max(losses)*0.02, f'{v:.4f}', ha='center', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Accuracy comparison
    axes[1].bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].set_ylim([0.8, 1.0])
    for i, v in enumerate(accuracies):
        axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Jaccard (IoU) comparison
    axes[2].bar(models, jaccards, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Jaccard Index (IoU)')
    axes[2].set_title('Validation Jaccard/IoU', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=15)
    for i, v in enumerate(jaccards):
        axes[2].text(i, v + max(jaccards)*0.02, f'{v:.4f}', ha='center', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(EXP_DIR / 'final_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: final_metrics_comparison.png")
    plt.close()

def plot_convergence_analysis(histories):
    """Analyze and plot convergence behavior"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Convergence Analysis - Validation Metrics', fontsize=16, fontweight='bold')

    colors = {'UNet': '#2E86AB', 'Attention UNet': '#A23B72', 'Attention ResUNet': '#F18F01'}

    # Validation Loss convergence
    ax = axes[0]
    for name, hist in histories.items():
        # Apply smoothing
        window = 10
        smoothed = hist['val_loss'].rolling(window=window, center=True).mean()
        ax.plot(smoothed, label=name, color=colors[name], linewidth=2.5)
        ax.scatter(hist['val_loss'].idxmin(), hist['val_loss'].min(),
                  color=colors[name], s=100, marker='*', edgecolor='black', linewidth=1.5,
                  zorder=5, label=f'{name} best')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss (smoothed)', fontsize=12)
    ax.set_title('Validation Loss Convergence', fontweight='bold', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Validation Jaccard convergence
    ax = axes[1]
    for name, hist in histories.items():
        # Apply smoothing
        window = 10
        smoothed = hist['val_jacard_coef'].rolling(window=window, center=True).mean()
        ax.plot(smoothed, label=name, color=colors[name], linewidth=2.5)
        ax.scatter(hist['val_jacard_coef'].idxmax(), hist['val_jacard_coef'].max(),
                  color=colors[name], s=100, marker='*', edgecolor='black', linewidth=1.5,
                  zorder=5, label=f'{name} best')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Jaccard (IoU, smoothed)', fontsize=12)
    ax.set_title('Validation Jaccard Convergence', fontweight='bold', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(EXP_DIR / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: convergence_analysis.png")
    plt.close()

def plot_overfitting_analysis(histories):
    """Analyze overfitting by comparing training vs validation metrics"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Overfitting Analysis - Training vs Validation Gap',
                 fontsize=16, fontweight='bold')

    colors = {'UNet': '#2E86AB', 'Attention UNet': '#A23B72', 'Attention ResUNet': '#F18F01'}

    for idx, (name, hist) in enumerate(histories.items()):
        ax = axes[idx]

        # Plot training and validation Jaccard
        epochs = range(len(hist))
        ax.plot(epochs, hist['jacard_coef'], label='Training Jaccard',
               color=colors[name], linewidth=2, alpha=0.8)
        ax.plot(epochs, hist['val_jacard_coef'], label='Validation Jaccard',
               color=colors[name], linewidth=2, linestyle='--', alpha=0.8)

        # Fill area between curves to show gap
        ax.fill_between(epochs, hist['jacard_coef'], hist['val_jacard_coef'],
                        alpha=0.2, color=colors[name])

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Jaccard Index (IoU)', fontsize=11)
        ax.set_title(f'{name}', fontweight='bold', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(EXP_DIR / 'overfitting_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: overfitting_analysis.png")
    plt.close()

def plot_training_time_comparison(summary):
    """Compare training time efficiency"""

    exec_times = summary['execution_times']
    models = ['UNet', 'Attention UNet', 'Attention ResUNet']

    # Convert time strings to minutes
    def time_to_minutes(time_str):
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 60 + minutes + seconds / 60

    times = [
        time_to_minutes(exec_times['unet']),
        time_to_minutes(exec_times['attention_unet']),
        time_to_minutes(exec_times['attention_resunet'])
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    bars = ax.bar(models, times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Training Time (minutes)', fontsize=12)
    ax.set_title('Training Time Comparison (200 epochs)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)

    # Add value labels on bars
    for bar, time, time_str in zip(bars, times, [exec_times['unet'],
                                                   exec_times['attention_unet'],
                                                   exec_times['attention_resunet']]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{time:.1f} min\n({time_str})',
               ha='center', va='bottom', fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(EXP_DIR / 'training_time_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: training_time_comparison.png")
    plt.close()

def calculate_statistics(histories, summary):
    """Calculate detailed statistics for the report"""

    stats = {}

    for name, hist in histories.items():
        model_key = name.lower().replace(' ', '_').replace('attention_', 'attention_')

        # Find best epoch
        best_epoch = hist['val_jacard_coef'].idxmax()

        # Calculate metrics at best epoch
        stats[name] = {
            'best_epoch': int(best_epoch),
            'best_val_jaccard': float(hist.loc[best_epoch, 'val_jacard_coef']),
            'best_val_accuracy': float(hist.loc[best_epoch, 'val_accuracy']),
            'best_val_loss': float(hist.loc[best_epoch, 'val_loss']),
            'final_val_jaccard': float(hist['val_jacard_coef'].iloc[-1]),
            'final_val_accuracy': float(hist['val_accuracy'].iloc[-1]),
            'final_val_loss': float(hist['val_loss'].iloc[-1]),
            'max_train_jaccard': float(hist['jacard_coef'].max()),
            'convergence_epoch': int(hist['val_loss'].idxmin()),
            'training_stability': float(hist['val_jacard_coef'].std())
        }

    return stats

def main():
    print("="*60)
    print("Xukuang Parameters Experiment Analysis")
    print("="*60)
    print()

    # Load data
    print("Loading experiment data...")
    exp_info, summary, histories = load_experiment_data()
    print(f"✓ Loaded data for {len(histories)} models")
    print()

    # Calculate statistics
    print("Calculating statistics...")
    stats = calculate_statistics(histories, summary)

    # Save statistics
    with open(EXP_DIR / 'analysis_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("✓ Saved: analysis_statistics.json")
    print()

    # Generate plots
    print("Generating visualizations...")
    plot_training_curves(histories, summary)
    plot_final_metrics_comparison(summary)
    plot_convergence_analysis(histories)
    plot_overfitting_analysis(histories)
    plot_training_time_comparison(summary)

    print()
    print("="*60)
    print("Analysis complete!")
    print("="*60)
    print()
    print("Summary:")
    print(f"  Best Model (by Jaccard): ", end="")
    best_model = max(stats.items(), key=lambda x: x[1]['final_val_jaccard'])
    print(f"{best_model[0]} (IoU: {best_model[1]['final_val_jaccard']:.4f})")
    print()
    for name in stats:
        print(f"  {name}:")
        print(f"    Final Val Jaccard: {stats[name]['final_val_jaccard']:.4f}")
        print(f"    Final Val Accuracy: {stats[name]['final_val_accuracy']:.4f}")
        print(f"    Best Val Jaccard: {stats[name]['best_val_jaccard']:.4f} (epoch {stats[name]['best_epoch']})")

if __name__ == "__main__":
    main()
