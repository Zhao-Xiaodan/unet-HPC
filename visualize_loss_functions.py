#!/usr/bin/env python3
"""
Visualize Loss Function Behavior
=================================
Create educational visualizations showing how Dice, Focal, and Combined losses
behave mathematically.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

OUTPUT_DIR = 'hyperparam_search_20251010_043123/analysis/'

def plot_loss_comparison():
    """Plot how each loss function behaves with prediction confidence"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Generate prediction probabilities
    p = np.linspace(0.01, 0.99, 200)

    # Ground truth scenarios
    y_positive = 1  # Foreground pixel
    y_negative = 0  # Background pixel

    # ==================== Panel 1: Dice Loss Behavior ====================
    ax = axes[0, 0]

    # For a single pixel, approximate Dice loss contribution
    # Simplified: assume |Y|=1, |Ŷ|=p for foreground, |Ŷ|=1-p for background
    dice_loss_fg = 1 - (2 * p) / (1 + p)  # Foreground pixel
    dice_loss_bg = 1 - (2 * (1-p)) / (1 + (1-p))  # Background pixel (inverted)

    ax.plot(p, dice_loss_fg, linewidth=3, label='Foreground (y=1)', color='#e74c3c')
    ax.plot(p, dice_loss_bg, linewidth=3, label='Background (y=0)', color='#3498db')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted Probability (p)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice Loss Contribution', fontsize=12, fontweight='bold')
    ax.set_title('Dice Loss: Treats All Pixels Equally', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.text(0.5, 0.95, 'Equal weight\nregardless of\nconfidence',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ==================== Panel 2: Focal Loss Behavior ====================
    ax = axes[0, 1]

    # Focal loss: FL(p) = -α * (1-p_t)^γ * log(p_t)
    alpha = 0.25
    gamma = 2.0

    # For foreground (y=1), p_t = p
    pt_fg = p
    focal_loss_fg = -alpha * np.power(1 - pt_fg, gamma) * np.log(np.clip(pt_fg, 1e-7, 1.0))

    # For background (y=0), p_t = 1-p
    pt_bg = 1 - p
    focal_loss_bg = -alpha * np.power(1 - pt_bg, gamma) * np.log(np.clip(pt_bg, 1e-7, 1.0))

    ax.plot(p, focal_loss_fg, linewidth=3, label='Foreground (y=1)', color='#e74c3c')
    ax.plot(p, focal_loss_bg, linewidth=3, label='Background (y=0)', color='#3498db')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted Probability (p)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Focal Loss', fontsize=12, fontweight='bold')
    ax.set_title('Focal Loss: Down-weights Easy Examples', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.text(0.5, 0.95, 'High penalty\nfor wrong\npredictions',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # ==================== Panel 3: Focusing Effect ====================
    ax = axes[1, 0]

    # Show the focusing factor (1-p_t)^γ
    pt = np.linspace(0.01, 0.99, 200)
    focusing_gamma1 = np.power(1 - pt, 1.0)  # γ=1 (no focusing)
    focusing_gamma2 = np.power(1 - pt, 2.0)  # γ=2 (our setting)
    focusing_gamma5 = np.power(1 - pt, 5.0)  # γ=5 (aggressive)

    ax.plot(pt, focusing_gamma1, linewidth=3, label='γ=1 (no focusing)',
            color='#95a5a6', linestyle='--')
    ax.plot(pt, focusing_gamma2, linewidth=3, label='γ=2 (our setting)',
            color='#e74c3c')
    ax.plot(pt, focusing_gamma5, linewidth=3, label='γ=5 (aggressive)',
            color='#c0392b', linestyle=':')

    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax.axhline(y=0.01, color='green', linestyle='--', alpha=0.7, linewidth=2)

    ax.set_xlabel('Confidence (p_t)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Focusing Factor (1-p_t)^γ', fontsize=12, fontweight='bold')
    ax.set_title('Focusing Mechanism: Down-weighting by Confidence',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Annotate key point
    ax.annotate('Easy example (p_t=0.9)\n99% weight reduction',
                xy=(0.9, 0.01), xytext=(0.6, 0.3),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=9, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # ==================== Panel 4: Gradient Comparison ====================
    ax = axes[1, 1]

    # Approximate gradients for foreground pixel (y=1)
    # Dice gradient (simplified): proportional to (2 - p) / (1+p)^2
    grad_dice = (2 - p) / np.power(1 + p, 2)

    # Focal gradient: α * γ * (1-p)^(γ-1) * (p - 1) / p (for y=1)
    grad_focal = alpha * gamma * np.power(1 - p, gamma - 1) * (1 - p) / np.clip(p, 1e-7, 1.0)

    # Combined gradient
    grad_combined = 0.7 * grad_dice + 0.3 * grad_focal

    ax.plot(p, grad_dice / grad_dice.max(), linewidth=3,
            label='Dice (70%)', color='#3498db', alpha=0.8)
    ax.plot(p, grad_focal / grad_focal.max(), linewidth=3,
            label='Focal (30%)', color='#e74c3c', alpha=0.8)
    ax.plot(p, grad_combined / grad_combined.max(), linewidth=4,
            label='Combined (0.7D + 0.3F)', color='#27ae60', linestyle='-')

    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted Probability (p)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Gradient Magnitude', fontsize=12, fontweight='bold')
    ax.set_title('Gradient Signals: Complementary Optimization',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)

    ax.text(0.05, 0.5, 'Low\nconfidence\nregion',
            transform=ax.transAxes, fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax.text(0.75, 0.5, 'High\nconfidence\nregion',
            transform=ax.transAxes, fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig6_loss_function_mathematics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created: fig6_loss_function_mathematics.png")

def plot_pixel_distribution():
    """Visualize pixel distribution and gradient influence at different resolutions"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ==================== Panel 1: Pixel Distribution ====================
    ax = axes[0]

    resolutions = ['256×256\n(65K pixels)', '512×512\n(262K pixels)']

    # Pixel counts (approximate for microbeads)
    background_256 = 50000
    foreground_256 = 13000
    boundary_256 = 2000

    background_512 = 230000
    foreground_512 = 28000
    boundary_512 = 4000

    x = np.arange(len(resolutions))
    width = 0.6

    # Stacked bar chart
    p1 = ax.bar(x, [background_256, background_512], width,
                label='Background (easy)', color='#ecf0f1', edgecolor='black')
    p2 = ax.bar(x, [foreground_256, foreground_512], width,
                bottom=[background_256, background_512],
                label='Foreground (medium)', color='#3498db', edgecolor='black')
    p3 = ax.bar(x, [boundary_256, boundary_512], width,
                bottom=[background_256+foreground_256, background_512+foreground_512],
                label='Boundary (hard)', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Number of Pixels', fontsize=12, fontweight='bold')
    ax.set_title('Pixel Distribution by Difficulty', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(resolutions, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Add percentage annotations
    total_256 = background_256 + foreground_256 + boundary_256
    total_512 = background_512 + foreground_512 + boundary_512

    ax.text(0, background_256/2, f'{100*background_256/total_256:.0f}%',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1, background_512/2, f'{100*background_512/total_512:.0f}%',
            ha='center', va='center', fontsize=10, fontweight='bold')

    ax.text(0, background_256 + foreground_256/2, f'{100*foreground_256/total_256:.0f}%',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.text(1, background_512 + foreground_512/2, f'{100*foreground_512/total_512:.0f}%',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    ax.text(0, background_256 + foreground_256 + boundary_256/2, f'{100*boundary_256/total_256:.0f}%',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.text(1, background_512 + foreground_512 + boundary_512/2, f'{100*boundary_512/total_512:.0f}%',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # ==================== Panel 2: Gradient Influence ====================
    ax = axes[1]

    categories = ['Without\nFocal Loss', 'With\nFocal Loss\n(γ=2)']

    # Gradient influence (relative)
    # Without focal: proportional to pixel count
    influence_background_none = 230000 / 262000
    influence_boundary_none = 4000 / 262000

    # With focal: background down-weighted by ~99%, boundary amplified
    influence_background_focal = (230000 * 0.01) / (230000 * 0.01 + 28000 * 0.1 + 4000 * 1.0)
    influence_boundary_focal = (4000 * 1.0) / (230000 * 0.01 + 28000 * 0.1 + 4000 * 1.0)

    x2 = np.arange(len(categories))

    # Side-by-side bars
    bar_width = 0.35
    bars1 = ax.bar(x2 - bar_width/2, [influence_boundary_none, influence_boundary_focal],
                   bar_width, label='Boundary pixels', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x2 + bar_width/2, [influence_background_none, influence_background_focal],
                   bar_width, label='Background pixels', color='#ecf0f1', edgecolor='black')

    ax.set_ylabel('Relative Gradient Influence', fontsize=12, fontweight='bold')
    ax.set_title('Gradient Influence on Optimization', fontsize=13, fontweight='bold')
    ax.set_xticks(x2)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add amplification annotation
    amplification = influence_boundary_focal / influence_boundary_none
    ax.annotate(f'{amplification:.0f}× amplification\nfor boundaries!',
                xy=(1, influence_boundary_focal),
                xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3),
                fontsize=11, color='#e74c3c', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#fadbd8', alpha=0.9))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig7_pixel_gradient_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created: fig7_pixel_gradient_analysis.png")

def main():
    """Generate all mathematical visualizations"""
    print("=" * 80)
    print("GENERATING MATHEMATICAL VISUALIZATIONS")
    print("=" * 80)
    print()

    plot_loss_comparison()
    plot_pixel_distribution()

    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files in {OUTPUT_DIR}:")
    print("  - fig6_loss_function_mathematics.png")
    print("  - fig7_pixel_gradient_analysis.png")
    print()

if __name__ == '__main__':
    main()
