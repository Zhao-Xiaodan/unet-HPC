# U-Net Hyperparameter Optimization Guide

## Overview

This package provides a comprehensive hyperparameter optimization framework for U-Net architectures in mitochondria segmentation, addressing the training dynamics issues identified in the original architecture comparison study.

## Files Included

### 1. Core Scripts
- **`pbs_hyperparameter_optimization.sh`** - Main PBS job script for grid search
- **`analyze_hyperparameter_results.py`** - Advanced results analysis script
- **`Hyperparameter_Optimization_Strategy.md`** - Detailed strategy document

### 2. Supporting Files
- **`CLAUDE.md`** - Environment configuration notes
- **Previous analysis results** in `mitochondria_segmentation_20250925_133928/`

## Quick Start

### Step 1: Prepare Environment

```bash
# Ensure conda environment is activated
source activate unetCNN

# Verify dataset structure
./pbs_setup_verification.sh
```

### Step 2: Submit Hyperparameter Optimization Job

```bash
# Submit the optimization job (48-hour runtime)
qsub pbs_hyperparameter_optimization.sh
```

### Step 3: Monitor Progress

```bash
# Check job status
qstat -u $USER

# Monitor progress (optional)
tail -f hyperparameter_optimization_*/exp_*/training_log.txt
```

### Step 4: Analyze Results

The analysis runs automatically at the end of the job, but you can also run it manually:

```bash
source activate unetCNN
python analyze_hyperparameter_results.py \
    --summary_file hyperparameter_optimization_*/hyperparameter_summary.csv \
    --output_dir hyperparameter_optimization_*/analysis
```

## Grid Search Configuration

### Hyperparameters Tested

| Parameter | Values | Rationale |
|-----------|---------|-----------|
| **Learning Rate** | [1e-4, 5e-4, 1e-3, 5e-3] | Address oscillation issues from original 1e-2 |
| **Batch Size** | [8, 16, 32] | Test gradient variance impact |
| **Architecture** | [UNet, Attention_UNet, Attention_ResUNet] | Fair comparison under stable conditions |
| **Epochs** | 30 | Sufficient for convergence assessment |

**Total Experiments:** 4 × 3 × 3 = 36 experiments
**Estimated Time:** 36-48 hours

### Enhanced Features

1. **Gradient Clipping** (clipnorm=1.0) - Prevents extreme weight updates
2. **Adaptive Learning Rate** - ReduceLROnPlateau with patience=5
3. **Improved Early Stopping** - Patience=10 for metric volatility
4. **Comprehensive Logging** - Individual experiment logs and results

## Expected Results

### Key Hypotheses

1. **Learning Rate Optimization**
   - Optimal range: 5e-4 to 1e-3
   - Standard U-Net benefits most from reduction
   - Attention mechanisms can handle slightly higher rates

2. **Batch Size Impact**
   - Larger batch sizes improve stability significantly
   - Effect most pronounced for Standard U-Net
   - Memory constraints may limit batch size 32

3. **Architecture-Specific Patterns**
   - Standard U-Net: Most hyperparameter sensitive
   - Attention U-Net: Benefits from larger batches after attention convergence
   - Attention ResU-Net: Most robust across configurations

## Output Files

### During Execution
```
hyperparameter_optimization_YYYYMMDD_HHMMSS/
├── exp_01_UNet_lr0.0001_bs8/
│   ├── training_log.txt
│   ├── UNet_lr0.0001_bs8_history.csv
│   ├── UNet_lr0.0001_bs8_model.hdf5
│   └── UNet_lr0.0001_bs8_results.json
├── exp_02_UNet_lr0.0001_bs16/
│   └── ...
└── hyperparameter_summary.csv
```

### Analysis Results
```
hyperparameter_optimization_YYYYMMDD_HHMMSS/
├── detailed_heatmaps.png                    # Performance heatmaps by architecture
├── comprehensive_analysis.png               # Comparative trend analysis
├── comprehensive_analysis_report.txt        # Detailed text report
├── analysis_summary.json                    # Machine-readable results
└── hyperparameter_summary.csv              # Raw results data
```

## Understanding the Results

### Key Metrics

1. **best_val_jaccard** - Primary performance metric (higher = better)
2. **val_loss_stability** - Training stability (lower = better)
3. **overfitting_gap** - Generalization assessment (val_loss - train_loss)
4. **epochs_to_best** - Convergence speed indicator

### Visualization Guide

1. **Detailed Heatmaps** - Performance and stability by (learning rate, batch size)
2. **Comparative Analysis** - Trends across hyperparameters with confidence intervals
3. **Architecture Comparison** - Box plots and scatter plots for trade-offs

### Statistical Analysis

- **ANOVA Tests** - Significance of hyperparameter effects
- **Effect Sizes** - Practical significance of differences
- **Correlation Analysis** - Relationships between metrics

## Troubleshooting

### Common Issues

1. **Memory Errors with Batch Size 32**
   ```bash
   # Check memory usage in logs
   grep "ResourceExhausted" exp_*/training_log.txt
   ```
   **Solution:** Experiments automatically skip if memory insufficient

2. **Failed Experiments**
   ```bash
   # Check failed experiments
   grep "False" hyperparameter_summary.csv
   ```
   **Solution:** Individual failures don't affect other experiments

3. **Container Issues**
   ```bash
   # Verify container exists
   ls -la /app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif
   ```
   **Solution:** Update container path in PBS script if needed

### Performance Issues

1. **Slow Convergence**
   - Check if learning rates too conservative (1e-4)
   - Consider extending epochs for final optimal configurations

2. **High Variance Results**
   - Increase number of random seeds in future experiments
   - Focus on batch size optimization

## Next Steps After Optimization

### Immediate Actions

1. **Rerun Original Comparison** with optimal hyperparameters
2. **Extended Training** (100+ epochs) for best configurations
3. **Statistical Validation** with multiple random seeds

### Advanced Analysis

1. **Transfer Learning** - Test on different datasets
2. **Ensemble Methods** - Combine multiple optimal configurations
3. **Architecture Modifications** - Based on hyperparameter insights

## Resource Requirements

### Computational Resources
- **GPU:** 1 × NVIDIA A40 (40GB VRAM)
- **CPU:** 36 cores for data loading parallelization
- **Memory:** 240GB system RAM
- **Storage:** ~50GB for all models and results

### Time Estimates
- **Grid Search:** 36-48 hours total
- **Per Experiment:** 60-90 minutes average
- **Analysis:** 30 minutes additional

## Citation and References

This hyperparameter optimization framework addresses the training dynamics analysis from:

> "U-Net Architecture Comparison for Mitochondria Segmentation: A Comprehensive Analysis of Training Dynamics and Performance Trade-offs"

Key insights implemented:
- Binary threshold sensitivity mitigation through learning rate reduction
- Gradient stability improvement through batch size optimization
- Architecture-specific hyperparameter recommendations

## Contact and Support

For questions or issues with the hyperparameter optimization:
1. Check logs in experiment directories
2. Review the comprehensive analysis report
3. Consult the strategy document for methodological details

---

**Version:** 1.0
**Last Updated:** September 2025
**Compatibility:** TensorFlow 2.16+, Python 3.8+