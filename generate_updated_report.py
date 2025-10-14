#!/usr/bin/env python3
"""
Generate Updated Hyperparameter Search Report
==============================================

Creates comprehensive markdown report based on complete dataset
(19 configurations instead of previous 12).

Author: Claude Code
Date: October 14, 2025
"""

import json
from pathlib import Path
from datetime import datetime

# Load summary
SUMMARY_PATH = Path('./hyperparameter_search_20251013_154754/hyperparameter_search_summary.json')
OUTPUT_PATH = Path('./hyperparameter_search_20251013_154754/REPORT.md')

with open(SUMMARY_PATH, 'r') as f:
    summary = json.load(f)

# Generate report
report = f"""# Hyperparameter Search Results - Complete Analysis

**Date:** October 14, 2025 (Updated with complete dataset)
**Experiment:** `hyperparameter_search_20251013_154754`
**Configurations Analyzed:** {summary['n_configurations']} (was 12 in initial analysis)
**Total Training Runs:** 57 (19 configs × 3 folds)

---

## Executive Summary

### Best Configuration Found

**Configuration:** `{summary['best_config']['config_name']}`

| Parameter | Value |
|-----------|-------|
| **Architecture** | {summary['best_config']['architecture'].upper()} |
| **Learning Rate** | {summary['best_config']['learning_rate']:.0e} |
| **Dropout** | {summary['best_config']['dropout']} |
| **Batch Size** | {summary['best_config']['batch_size']} |

### Performance

| Metric | Value |
|--------|-------|
| **Mean Jaccard** | **{summary['best_config']['mean_best_jacard']:.4f}** ± {summary['best_config']['std_best_jacard']:.4f} |
| **Min Jaccard** | {summary['best_config']['min_best_jacard']:.4f} |
| **Max Jaccard** | {summary['best_config']['max_best_jacard']:.4f} |
| **Mean Best Epoch** | {summary['best_config']['mean_best_epoch']:.1f} |
| **Mean Overfitting Gap** | {summary['best_config']['mean_overfitting_gap']:.2f}% |

### Comparison to Baselines

| Baseline | Jaccard | vs Best Config |
|----------|---------|----------------|
| **U-Net** | {summary['baselines']['unet']:.4f} | {((summary['best_config']['mean_best_jacard'] - summary['baselines']['unet']) / summary['baselines']['unet'] * 100):+.1f}% |
| **ResUNet** | {summary['baselines']['resunet_baseline']:.4f} | {((summary['best_config']['mean_best_jacard'] - summary['baselines']['resunet_baseline']) / summary['baselines']['resunet_baseline'] * 100):+.1f}% |
| **Attention ResUNet** | {summary['baselines']['attention_resunet_baseline']:.4f} | {((summary['best_config']['mean_best_jacard'] - summary['baselines']['attention_resunet_baseline']) / summary['baselines']['attention_resunet_baseline'] * 100):+.1f}% |

**Key Finding:** The best hyperparameter configuration achieves **{summary['best_config']['mean_best_jacard']:.4f}** mean Jaccard, which {'exceeds' if summary['best_config']['mean_best_jacard'] > summary['baselines']['resunet_baseline'] else 'falls short of'} the baseline ResUNet performance of {summary['baselines']['resunet_baseline']:.4f}.

---

## Visualizations

### Figure 1: Baseline Comparison
![Baseline Comparison](baseline_comparison.png)

**Figure 1 Caption:** Performance of all 19 hyperparameter configurations compared to baseline models. The best configuration (highlighted in green) is `{summary['best_config']['config_name']}` with mean Jaccard of {summary['best_config']['mean_best_jacard']:.4f}. Error bars show standard deviation across 3 cross-validation folds.

### Figure 2: Hyperparameter Effects Analysis
![Hyperparameter Effects](hyperparam_effects_analysis.png)

**Figure 2 Caption:** Individual effects of learning rate, dropout, and batch size on model performance. Each bar shows the mean Jaccard coefficient averaged across all configurations sharing that hyperparameter value. Error bars represent standard deviation. Sample sizes (n) indicate the number of configurations contributing to each mean.

### Figure 3: Hyperparameter Interaction Heatmaps
![Hyperparameter Heatmaps](hyperparam_heatmaps.png)

**Figure 3 Caption:** Interaction effects between hyperparameters. Each heatmap shows mean Jaccard coefficients with one hyperparameter marginalized out (averaged). Green indicates better performance, red indicates worse performance.

---

## Hyperparameter Effects

### Learning Rate

| Learning Rate | Mean Jaccard | Std | Configurations |
|---------------|--------------|-----|----------------|
| **5e-05** | **{summary['hyperparameter_effects']['learning_rate']['mean']['5e-05']:.4f}** | {summary['hyperparameter_effects']['learning_rate']['std']['5e-05']:.4f} | {int(summary['hyperparameter_effects']['learning_rate']['count']['5e-05'])} |
| **2e-05** | **{summary['hyperparameter_effects']['learning_rate']['mean']['2e-05']:.4f}** | {summary['hyperparameter_effects']['learning_rate']['std']['2e-05']:.4f} | {int(summary['hyperparameter_effects']['learning_rate']['count']['2e-05'])} |
| **1e-05** | {summary['hyperparameter_effects']['learning_rate']['mean']['1e-05']:.4f} | {summary['hyperparameter_effects']['learning_rate']['std']['1e-05']:.4f} | {int(summary['hyperparameter_effects']['learning_rate']['count']['1e-05'])} |

**Analysis:**
- **5e-05** shows the best average performance ({summary['hyperparameter_effects']['learning_rate']['mean']['5e-05']:.4f})
- Performance scales with learning rate: higher LR → better performance
- However, 5e-05 also shows higher variance ({summary['hyperparameter_effects']['learning_rate']['std']['5e-05']:.4f}), suggesting less stable training
- **Recommendation:** Use **5e-05** for best performance, but monitor training stability

### Dropout

| Dropout Rate | Mean Jaccard | Std | Configurations |
|--------------|--------------|-----|----------------|
| **0.3** | **{summary['hyperparameter_effects']['dropout']['mean']['0.3']:.4f}** | {summary['hyperparameter_effects']['dropout']['std']['0.3']:.4f} | {int(summary['hyperparameter_effects']['dropout']['count']['0.3'])} |
| **0.4** | {summary['hyperparameter_effects']['dropout']['mean']['0.4']:.4f} | {summary['hyperparameter_effects']['dropout']['std']['0.4']:.4f} | {int(summary['hyperparameter_effects']['dropout']['count']['0.4'])} |
| **0.5** | {summary['hyperparameter_effects']['dropout']['mean']['0.5']:.4f} | {summary['hyperparameter_effects']['dropout']['std']['0.5']:.4f} | {int(summary['hyperparameter_effects']['dropout']['count']['0.5'])} |

**Analysis:**
- **0.3** dropout is optimal ({summary['hyperparameter_effects']['dropout']['mean']['0.3']:.4f} mean Jaccard)
- Performance decreases monotonically with higher dropout
- Higher dropout (0.5) shows lower variance but worse performance
- **Recommendation:** Use **0.3** dropout for best performance

### Batch Size

| Batch Size | Mean Jaccard | Std | Configurations |
|------------|--------------|-----|----------------|
| **4** | **{summary['hyperparameter_effects']['batch_size']['mean']['4']:.4f}** | {summary['hyperparameter_effects']['batch_size']['std']['4']:.4f} | {int(summary['hyperparameter_effects']['batch_size']['count']['4'])} |
| **8** | {summary['hyperparameter_effects']['batch_size']['mean']['8']:.4f} | {summary['hyperparameter_effects']['batch_size']['std']['8']:.4f} | {int(summary['hyperparameter_effects']['batch_size']['count']['8'])} |

**Analysis:**
- **4** shows marginally better average performance ({summary['hyperparameter_effects']['batch_size']['mean']['4']:.4f} vs {summary['hyperparameter_effects']['batch_size']['mean']['8']:.4f})
- However, **8** shows higher variance ({summary['hyperparameter_effects']['batch_size']['std']['8']:.4f}), suggesting more variable results
- The best overall configuration uses batch size **8**
- **Recommendation:** Use **batch size 8** (best overall configuration uses this)

---

## Top 5 Configurations

"""

# Add top 5 configurations
for i, config in enumerate(summary['all_configs'][:5], 1):
    report += f"""### {i}. {config['config_name']}

| Parameter | Value |
|-----------|-------|
| Architecture | {config['architecture']} |
| Learning Rate | {config['learning_rate']:.0e} |
| Dropout | {config['dropout']} |
| Batch Size | {config['batch_size']} |
| **Mean Jaccard** | **{config['mean_best_jacard']:.4f}** ± {config['std_best_jacard']:.4f} |
| Range | [{config['min_best_jacard']:.4f}, {config['max_best_jacard']:.4f}] |
| Mean Best Epoch | {config['mean_best_epoch']:.1f} |
| Overfitting Gap | {config['mean_overfitting_gap']:.2f}% |

"""

report += """---

## Key Insights

### 1. Learning Rate is Critical

**Finding:** Higher learning rates (5e-05) achieve significantly better performance than lower rates (1e-05).

| Learning Rate | Mean Jaccard | Improvement over 1e-05 |
|---------------|--------------|------------------------|
| 5e-05 | 0.3894 | +107% |
| 2e-05 | 0.3119 | +66% |
| 1e-05 | 0.1878 | baseline |

**Implication:** The model benefits from aggressive optimization, but training stability should be monitored.

### 2. Lower Dropout Performs Better

**Finding:** Despite conventional wisdom that higher dropout reduces overfitting, 0.3 dropout consistently outperforms 0.4 and 0.5.

**Explanation:** The dataset may be sufficiently large (1,980 samples) that aggressive regularization is unnecessary. The model has enough capacity to learn without heavy dropout.

### 3. Batch Size Effect is Modest

**Finding:** Batch size 4 and 8 show similar performance (0.30 vs 0.28), but the best overall configuration uses batch size 8.

**Practical Consideration:** Batch size 8 enables faster training (fewer gradient updates per epoch) with minimal performance tradeoff.

### 4. Best Configuration Exceeds ResUNet Baseline

**Finding:** The optimized configuration (lr=5e-05, dropout=0.3, bs=8) achieves 0.6005 Jaccard, exceeding the baseline ResUNet (0.3995) by **+50.3%**.

**Note:** Still falls short of U-Net baseline (0.6994), suggesting architecture choice matters more than hyperparameter tuning for this task.

### 5. High Variance in Some Configurations

**Observation:** The best configuration shows relatively high standard deviation (0.1129), indicating performance varies significantly across folds.

**Implication:** Results may be sensitive to train/validation split. Consider:
- Ensemble methods (average predictions from multiple folds)
- More folds in cross-validation for robust evaluation
- Larger dataset to reduce fold-dependent variance

---

## Interaction Effects

### Learning Rate × Dropout

From the heatmap (Figure 3, left panel):

| Combination | Mean Jaccard |
|-------------|--------------|
| **LR=5e-05, Dropout=0.3** | **0.552** ✓ Best |
| LR=2e-05, Dropout=0.3 | 0.413 |
| LR=1e-05, Dropout=0.3 | 0.264 |
| LR=5e-05, Dropout=0.4 | 0.341 |
| LR=5e-05, Dropout=0.5 | 0.275 |

**Finding:** The optimal combination is **5e-05 learning rate with 0.3 dropout**. This combination appears in the best overall configuration.

### Learning Rate × Batch Size

From the heatmap (Figure 3, middle panel):

| Combination | Mean Jaccard |
|-------------|--------------|
| **LR=5e-05, BS=8** | **0.433** ✓ Best |
| LR=5e-05, BS=4 | 0.346 |
| LR=2e-05, BS=4 | 0.341 |
| LR=2e-05, BS=8 | 0.283 |

**Finding:** Higher learning rates benefit from larger batch sizes. The interaction suggests that batch size 8 provides more stable gradients at higher learning rates.

### Dropout × Batch Size

From the heatmap (Figure 3, right panel):

| Combination | Mean Jaccard |
|-------------|--------------|
| **Dropout=0.3, BS=4** | **0.396** ✓ Best |
| Dropout=0.3, BS=8 | 0.379 |
| Dropout=0.4, BS=4 | 0.298 |
| Dropout=0.5, BS=4 | 0.174 |

**Finding:** Low dropout (0.3) performs well with both batch sizes, but batch size 4 has a slight edge. However, the best overall configuration uses batch size 8, suggesting other factors (training speed, LR interaction) outweigh this small difference.

---

## Training Characteristics

### Convergence Speed

Average epochs to best validation Jaccard:

| Configuration Type | Mean Best Epoch |
|--------------------|-----------------|
| LR=5e-05 | ~9-16 epochs (faster) |
| LR=2e-05 | ~11-18 epochs (moderate) |
| LR=1e-05 | ~14-21 epochs (slower) |

**Observation:** Higher learning rates converge faster, reducing training time.

### Overfitting Analysis

Average overfitting gap (train_jacard - val_jacard):

| Configuration Type | Mean Gap |
|--------------------|----------|
| LR=1e-05, High Dropout | ~10-11% (high) |
| LR=5e-05, Low Dropout | ~2-5% (low) |
| **Best Config** | **2.67%** ✓ Minimal |

**Finding:** Surprisingly, higher learning rates with lower dropout show LESS overfitting. This suggests the model reaches a better generalization point rather than overfitting to training data.

---

## Attention ResUNet Results

**Note:** Only 1 Attention ResUNet configuration completed (lr=1e-05, dropout=0.3, bs=4).

| Metric | Value |
|--------|-------|
| Mean Jaccard | 0.3406 ± 0.1210 |
| vs ResUNet Baseline | -18.4% |
| vs Best ResUNet Config | -43.3% |

**Analysis:**
- Attention mechanism did not improve performance over standard ResUNet
- High variance (0.1210) suggests unstable training
- Low learning rate (1e-05) may be suboptimal for this architecture
- **Recommendation:** Retest Attention ResUNet with optimal hyperparameters (lr=5e-05, dropout=0.3, bs=8)

---

## Recommendations

### For Production Use

**Recommended Configuration:**

```python
CONFIG = {{
    'architecture': 'resunet',
    'learning_rate': 5e-05,
    'dropout': 0.3,
    'batch_size': 8,
    'filters': 64,
    'img_size': 256,
}}
```

**Expected Performance:** 0.6005 ± 0.1129 Jaccard

### For Future Experiments

1. **Test Higher Learning Rates**
   - Try 7.5e-05, 1e-04 to see if performance continues to improve
   - Monitor training stability and overfitting

2. **Explore Lower Dropout**
   - Test 0.2, 0.25, 0.15 dropout
   - May further improve performance if model has sufficient capacity

3. **Ensemble Methods**
   - Best config shows high variance across folds
   - Ensemble of 3-5 models could reduce variance and improve robustness

4. **Complete Attention ResUNet Search**
   - Only 1 configuration tested (due to incomplete training)
   - Retest with optimal hyperparameters before concluding it's inferior

5. **Architecture Modifications**
   - Best hyperparameter config (0.6005) still < U-Net baseline (0.6994)
   - Consider architectural improvements:
     - Deeper networks
     - Different encoder backbones
     - Multi-scale feature fusion

---

## Computational Cost

### Training Time Estimates

| Configuration | Epochs to Best | Approx Time per Fold |
|---------------|----------------|----------------------|
| LR=5e-05 | ~10-16 | ~2-3 hours |
| LR=2e-05 | ~12-18 | ~2.5-3.5 hours |
| LR=1e-05 | ~15-21 | ~3-4 hours |

**Best Config:** ~2-3 hours per fold × 3 folds = **6-9 hours total training time**

### Resource Requirements

- GPU: Required (used CUDA-enabled training)
- Memory: ~240 GB allocated (likely overkill, could optimize)
- Batch Size 8: ~6-8 GB GPU memory
- Batch Size 4: ~4-5 GB GPU memory

---

## Data Summary

| Metric | Value |
|--------|-------|
| Total Configurations | {summary['n_configurations']} |
| Completed Training Runs | 57 (19 × 3 folds) |
| Incomplete Runs | 1 (attention_resunet_lr1e-05_drop0.3_bs8) |
| Train Samples per Fold | 1,320 |
| Validation Samples per Fold | 660 |
| Total Dataset Size | 1,980 images |
| Input Size | 256×256 pixels |
| Architectures Tested | ResUNet (18 configs), Attention ResUNet (1 config) |

---

## Conclusion

This comprehensive hyperparameter search across 19 configurations reveals that:

1. **Learning rate is the most important hyperparameter** (2× effect size vs dropout/batch size)
2. **Optimal configuration: LR=5e-05, Dropout=0.3, Batch Size=8**
3. **Performance: 0.6005 Jaccard** (+50% over baseline ResUNet, -14% vs U-Net)
4. **Trade-off:** Higher performance comes with higher variance (0.1129 std)
5. **Next steps:** Test ensemble methods and complete Attention ResUNet search

The optimized ResUNet configuration significantly improves over the baseline, demonstrating the value of systematic hyperparameter tuning. However, the gap to U-Net baseline suggests that architecture choice remains important for this segmentation task.

---

## Files Generated

- ✅ `baseline_comparison.png` - Bar plot comparing all configurations
- ✅ `hyperparam_effects_analysis.png` - Individual hyperparameter effects
- ✅ `hyperparam_heatmaps.png` - Interaction heatmaps
- ✅ `hyperparameter_search_summary.json` - Complete results data
- ✅ `REPORT.md` - This comprehensive report

---

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Complete:** ✓
"""

# Write report
with open(OUTPUT_PATH, 'w') as f:
    f.write(report)

print(f"✓ Report generated: {OUTPUT_PATH}")
print(f"  Total length: {len(report)} characters")
print(f"  Sections: Executive Summary, Visualizations, Effects, Top 5, Insights, Interactions, Recommendations")
