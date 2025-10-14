#!/usr/bin/env python3
"""
Comprehensive analysis of microscope dataset training results
Generates additional visualizations and detailed statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Directory containing results
results_dir = 'microscope_training_20251008_074915'

# Read training histories
unet_df = pd.read_csv(f'{results_dir}/unet_history.csv')
att_unet_df = pd.read_csv(f'{results_dir}/attention_unet_history.csv')
att_res_unet_df = pd.read_csv(f'{results_dir}/attention_resunet_history.csv')

print("=" * 80)
print("MICROSCOPE DATASET TRAINING ANALYSIS")
print("=" * 80)
print()

# Summary statistics
models = {
    'UNet': unet_df,
    'Attention_UNet': att_unet_df,
    'Attention_ResUNet': att_res_unet_df
}

summary_data = []
for model_name, df in models.items():
    best_epoch = df['val_jacard_coef'].idxmax()
    summary_data.append({
        'Model': model_name,
        'Total_Epochs': len(df),
        'Best_Epoch': best_epoch + 1,
        'Best_Val_Jaccard': df['val_jacard_coef'].iloc[best_epoch],
        'Final_Val_Jaccard': df['val_jacard_coef'].iloc[-1],
        'Best_Val_Loss': df['val_loss'].iloc[best_epoch],
        'Final_Val_Loss': df['val_loss'].iloc[-1],
        'Best_Train_Jaccard': df['jacard_coef'].iloc[best_epoch],
        'Final_Train_Jaccard': df['jacard_coef'].iloc[-1],
        'Convergence_Stability': df['val_loss'].iloc[-10:].std(),
        'Learning_Rate': df['lr'].iloc[0]
    })

summary_df = pd.DataFrame(summary_data)
print("SUMMARY STATISTICS")
print("-" * 80)
print(summary_df.to_string(index=False))
print()

# Additional analysis
print("DETAILED ANALYSIS")
print("-" * 80)
for model_name, df in models.items():
    print(f"\n{model_name}:")
    print(f"  Epochs trained: {len(df)}")
    print(f"  Best validation Jaccard: {df['val_jacard_coef'].max():.6f} (epoch {df['val_jacard_coef'].idxmax() + 1})")
    print(f"  Final validation Jaccard: {df['val_jacard_coef'].iloc[-1]:.6f}")
    print(f"  Improvement over training: {(df['val_jacard_coef'].max() - df['val_jacard_coef'].iloc[0]):.6f}")
    print(f"  Final train-val gap: {(df['jacard_coef'].iloc[-1] - df['val_jacard_coef'].iloc[-1]):.6f}")
    print(f"  Convergence stability (last 10 epochs): {df['val_loss'].iloc[-10:].std():.6f}")

print()

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Row 1: Jaccard coefficient evolution
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(unet_df.index + 1, unet_df['jacard_coef'], 'b-', label='Train', alpha=0.7)
ax1.plot(unet_df.index + 1, unet_df['val_jacard_coef'], 'r-', label='Val', alpha=0.7)
ax1.set_title('UNet - Jaccard Coefficient', fontweight='bold', fontsize=11)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Jaccard Coefficient')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(att_unet_df.index + 1, att_unet_df['jacard_coef'], 'b-', label='Train', alpha=0.7)
ax2.plot(att_unet_df.index + 1, att_unet_df['val_jacard_coef'], 'r-', label='Val', alpha=0.7)
ax2.set_title('Attention UNet - Jaccard Coefficient', fontweight='bold', fontsize=11)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Jaccard Coefficient')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(att_res_unet_df.index + 1, att_res_unet_df['jacard_coef'], 'b-', label='Train', alpha=0.7)
ax3.plot(att_res_unet_df.index + 1, att_res_unet_df['val_jacard_coef'], 'r-', label='Val', alpha=0.7)
ax3.set_title('Attention ResUNet - Jaccard Coefficient', fontweight='bold', fontsize=11)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Jaccard Coefficient')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Comparison plot
ax4 = fig.add_subplot(gs[0, 3])
ax4.plot(unet_df.index + 1, unet_df['val_jacard_coef'], label='UNet', linewidth=2)
ax4.plot(att_unet_df.index + 1, att_unet_df['val_jacard_coef'], label='Att UNet', linewidth=2)
ax4.plot(att_res_unet_df.index + 1, att_res_unet_df['val_jacard_coef'], label='Att ResUNet', linewidth=2)
ax4.set_title('Validation Jaccard Comparison', fontweight='bold', fontsize=11)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Val Jaccard Coefficient')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Row 2: Loss curves
ax5 = fig.add_subplot(gs[1, 0])
ax5.plot(unet_df.index + 1, unet_df['loss'], 'b-', label='Train', alpha=0.7)
ax5.plot(unet_df.index + 1, unet_df['val_loss'], 'r-', label='Val', alpha=0.7)
ax5.set_title('UNet - Loss', fontweight='bold', fontsize=11)
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Loss')
ax5.set_yscale('log')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[1, 1])
ax6.plot(att_unet_df.index + 1, att_unet_df['loss'], 'b-', label='Train', alpha=0.7)
ax6.plot(att_unet_df.index + 1, att_unet_df['val_loss'], 'r-', label='Val', alpha=0.7)
ax6.set_title('Attention UNet - Loss', fontweight='bold', fontsize=11)
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Loss')
ax6.set_yscale('log')
ax6.legend()
ax6.grid(True, alpha=0.3)

ax7 = fig.add_subplot(gs[1, 2])
ax7.plot(att_res_unet_df.index + 1, att_res_unet_df['loss'], 'b-', label='Train', alpha=0.7)
ax7.plot(att_res_unet_df.index + 1, att_res_unet_df['val_loss'], 'r-', label='Val', alpha=0.7)
ax7.set_title('Attention ResUNet - Loss', fontweight='bold', fontsize=11)
ax7.set_xlabel('Epoch')
ax7.set_ylabel('Loss')
ax7.set_yscale('log')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Loss comparison
ax8 = fig.add_subplot(gs[1, 3])
ax8.plot(unet_df.index + 1, unet_df['val_loss'], label='UNet', linewidth=2)
ax8.plot(att_unet_df.index + 1, att_unet_df['val_loss'], label='Att UNet', linewidth=2)
ax8.plot(att_res_unet_df.index + 1, att_res_unet_df['val_loss'], label='Att ResUNet', linewidth=2)
ax8.set_title('Validation Loss Comparison', fontweight='bold', fontsize=11)
ax8.set_xlabel('Epoch')
ax8.set_ylabel('Val Loss (log scale)')
ax8.set_yscale('log')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Row 3: Performance metrics
ax9 = fig.add_subplot(gs[2, 0])
metrics = summary_df['Model'].values
best_jaccards = summary_df['Best_Val_Jaccard'].values
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax9.bar(metrics, best_jaccards, color=colors, alpha=0.7, edgecolor='black')
ax9.set_title('Best Validation Jaccard', fontweight='bold', fontsize=11)
ax9.set_ylabel('Jaccard Coefficient')
ax9.set_ylim([0, max(best_jaccards) * 1.2])
for i, (bar, val) in enumerate(zip(bars, best_jaccards)):
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
ax9.grid(True, alpha=0.3, axis='y')

ax10 = fig.add_subplot(gs[2, 1])
epochs_to_best = summary_df['Best_Epoch'].values
bars = ax10.bar(metrics, epochs_to_best, color=colors, alpha=0.7, edgecolor='black')
ax10.set_title('Epochs to Best Performance', fontweight='bold', fontsize=11)
ax10.set_ylabel('Epoch Number')
for i, (bar, val) in enumerate(zip(bars, epochs_to_best)):
    ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
ax10.grid(True, alpha=0.3, axis='y')

ax11 = fig.add_subplot(gs[2, 2])
stability = summary_df['Convergence_Stability'].values
bars = ax11.bar(metrics, stability, color=colors, alpha=0.7, edgecolor='black')
ax11.set_title('Training Stability (last 10 epochs)', fontweight='bold', fontsize=11)
ax11.set_ylabel('Val Loss Std Dev')
ax11.set_yscale('log')
for i, (bar, val) in enumerate(zip(bars, stability)):
    ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
ax11.grid(True, alpha=0.3, axis='y')

# Train-val gap analysis
ax12 = fig.add_subplot(gs[2, 3])
train_val_gap = summary_df['Final_Train_Jaccard'].values - summary_df['Final_Val_Jaccard'].values
bars = ax12.bar(metrics, train_val_gap, color=colors, alpha=0.7, edgecolor='black')
ax12.set_title('Train-Val Jaccard Gap (Final)', fontweight='bold', fontsize=11)
ax12.set_ylabel('Jaccard Difference')
ax12.axhline(y=0, color='black', linestyle='--', linewidth=1)
for i, (bar, val) in enumerate(zip(bars, train_val_gap)):
    ax12.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if val > 0 else bar.get_height() - 0.02,
             f'{val:.4f}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold', fontsize=9)
ax12.grid(True, alpha=0.3, axis='y')

plt.suptitle('Comprehensive Training Analysis - Microscope Dataset',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(f'{results_dir}/detailed_training_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved detailed analysis: {results_dir}/detailed_training_analysis.png")
plt.close()

# Create epoch-by-epoch comparison table
comparison_data = []
max_epochs = max(len(unet_df), len(att_unet_df), len(att_res_unet_df))

for epoch in range(max_epochs):
    row = {'Epoch': epoch + 1}
    if epoch < len(unet_df):
        row['UNet_Val_Jaccard'] = unet_df['val_jacard_coef'].iloc[epoch]
        row['UNet_Val_Loss'] = unet_df['val_loss'].iloc[epoch]
    if epoch < len(att_unet_df):
        row['AttUNet_Val_Jaccard'] = att_unet_df['val_jacard_coef'].iloc[epoch]
        row['AttUNet_Val_Loss'] = att_unet_df['val_loss'].iloc[epoch]
    if epoch < len(att_res_unet_df):
        row['AttResUNet_Val_Jaccard'] = att_res_unet_df['val_jacard_coef'].iloc[epoch]
        row['AttResUNet_Val_Loss'] = att_res_unet_df['val_loss'].iloc[epoch]
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(f'{results_dir}/epoch_by_epoch_comparison.csv', index=False)
print(f"Saved epoch comparison: {results_dir}/epoch_by_epoch_comparison.csv")

# Save summary statistics
summary_df.to_csv(f'{results_dir}/training_summary_statistics.csv', index=False)
print(f"Saved summary statistics: {results_dir}/training_summary_statistics.csv")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
