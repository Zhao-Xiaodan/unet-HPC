#!/usr/bin/env python3
"""
Analyze 512×512 Hyperparameter Search Results
==============================================

Analyzes results from hyperparameter_search_512_20251014_142259/
Compares with 256×256 results and generates comprehensive report.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load results
results_dir = Path('hyperparameter_search_512_20251014_142259')
df = pd.read_csv(results_dir / 'all_results.csv')

print("="*80)
print("512×512 HYPERPARAMETER SEARCH RESULTS ANALYSIS")
print("="*80)
print(f"Results directory: {results_dir}")
print(f"Timestamp: 2025-10-14 14:22:59 (HPC run)")
print()

# Overall statistics
print("="*80)
print("OVERALL STATISTICS")
print("="*80)
print(f"Total configurations tested: {df['config_name'].nunique()}")
print(f"Total training runs (configs × folds): {len(df)}")
print(f"Successful runs: {df['success'].sum()}/{len(df)} ({df['success'].mean()*100:.1f}%)")
print(f"OOM failures: {(~df['success']).sum()} ({(~df['success']).mean()*100:.1f}%)")
print()

# Best configuration
successful = df[df['success'] == True]
best_config = successful.groupby('config_name')['val_jaccard'].mean().idxmax()
best_jaccard = successful.groupby('config_name')['val_jaccard'].mean().max()

print("="*80)
print("BEST CONFIGURATION")
print("="*80)
print(f"Configuration: {best_config}")
print(f"Mean Jaccard: {best_jaccard:.4f}")
print()

# Parse best config
parts = best_config.split('_')
best_arch = parts[0]
best_lr = parts[1].replace('lr', '')
best_dropout = float(parts[2].replace('drop', ''))
best_bs = int(parts[3].replace('bs', ''))

print(f"  Architecture: {best_arch}")
print(f"  Learning Rate: {best_lr}")
print(f"  Dropout: {best_dropout}")
print(f"  Batch Size: {best_bs}")
print()

# Breakdown by architecture
print("="*80)
print("PERFORMANCE BY ARCHITECTURE")
print("="*80)
arch_summary = successful.groupby('architecture')['val_jaccard'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max'),
    ('count', 'count')
])
print(arch_summary.round(4))
print()

# Breakdown by hyperparameters
print("="*80)
print("PERFORMANCE BY HYPERPARAMETER")
print("="*80)

print("\nLearning Rate:")
lr_summary = successful.groupby('lr')['val_jaccard'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('count', 'count')
]).sort_values('mean', ascending=False)
print(lr_summary.round(4))

print("\nDropout:")
dropout_summary = successful.groupby('dropout')['val_jaccard'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('count', 'count')
]).sort_values('mean', ascending=False)
print(dropout_summary.round(4))

print("\nBatch Size:")
bs_summary = successful.groupby('batch_size')['val_jaccard'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('count', 'count')
]).sort_values('mean', ascending=False)
print(bs_summary.round(4))

# Top configurations
print("\n" + "="*80)
print("TOP 10 CONFIGURATIONS")
print("="*80)
top_configs = successful.groupby('config_name')['val_jaccard'].agg([
    ('mean', 'mean'),
    ('std', 'std')
]).sort_values('mean', ascending=False).head(10)
print(top_configs.round(4))

# Training characteristics
print("\n" + "="*80)
print("TRAINING CHARACTERISTICS")
print("="*80)
print(f"Mean best epoch: {successful['best_epoch'].mean():.1f}")
print(f"Mean total epochs: {successful['total_epochs'].mean():.1f}")
print(f"Mean overfitting gap: {successful['overfitting_gap'].mean():.2f}%")
print()
print(f"Mean batch size used: {successful['batch_size_used'].mean():.1f}")
print(f"Batch size reductions (OOM retries > 0): {(successful['oom_retries'] > 0).sum()} runs")

# Comparison with 256×256 (if available)
print("\n" + "="*80)
print("COMPARISON WITH 256×256 RESULTS")
print("="*80)

print("\n256×256 Best Results (from previous experiments):")
print("  Best config: resunet_lr5e-05_drop0.3_bs8")
print("  Best Jaccard: 0.6005")
print()

print("512×512 Results (current):")
print(f"  Best config: {best_config}")
print(f"  Best Jaccard: {best_jaccard:.4f}")
print()

jaccard_ratio = best_jaccard / 0.6005
print(f"Performance Ratio (512÷256): {jaccard_ratio:.2%}")
print()

if best_jaccard < 0.6005:
    diff = 0.6005 - best_jaccard
    print(f"⚠ WARNING: 512×512 performance is {diff:.4f} LOWER than 256×256!")
    print()
    print("Possible reasons:")
    print("  1. Reduced filters (32 vs 64) - Less model capacity")
    print("  2. Smaller dataset (98 images vs 1,980 patches)")
    print("  3. RGB vs Grayscale (different data distribution)")
    print("  4. Training instability (loss=nan observed in logs)")
else:
    diff = best_jaccard - 0.6005
    print(f"✓ 512×512 performance is {diff:.4f} HIGHER than 256×256!")

# Warning about training issues
print("\n" + "="*80)
print("TRAINING ISSUES DETECTED")
print("="*80)
print("⚠ CRITICAL: Loss values are 'nan' in training logs!")
print()
print("Evidence from logs:")
print("  - loss: nan - jacard_coef: 0.1226")
print("  - val_loss: nan - val_jacard_coef: 0.1613")
print()
print("This indicates:")
print("  1. Numerical instability during training")
print("  2. Gradient explosion or vanishing")
print("  3. Mixed precision issues")
print()
print("Despite nan loss, Jaccard metrics were computed successfully.")
print("Results may be unreliable due to unstable training.")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\n1. Fix Training Instability:")
print("   - Disable mixed precision (use FP32 instead of FP16)")
print("   - Add gradient clipping")
print("   - Reduce learning rate further")
print()

print("2. Improve Model Capacity:")
print("   - Increase filters from 32 to 48 or 64")
print("   - May require reducing batch size to fit in memory")
print()

print("3. Dataset Considerations:")
print("   - 98 images is very small for 512×512 training")
print("   - Consider data augmentation")
print("   - Or use 256×256 models with upscaling")
print()

print("4. Architecture Selection:")
print(f"   - Best architecture: {best_arch}")
print("   - Consider focusing future experiments on this architecture")
print()

print("5. Next Steps:")
print("   a) Re-run with fixed training (no nan loss)")
print("   b) If results remain low, use 256×256 models instead")
print("   c) For production: 256×256 models are more reliable")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
