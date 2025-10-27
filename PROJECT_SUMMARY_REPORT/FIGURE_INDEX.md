# Figure Index for Project Summary Report

This directory contains all figures referenced in the comprehensive project summary report. Figures have been copied from their original experiment folders to ensure they remain accessible even if those folders are moved or deleted.

## Figure List

### Training and Architecture Comparison

**From xukuang_params_shrunk_20251015_071224/:**
- `training_curves_comparison.png` - Training curves for UNet, Attention UNet, and Attention ResUNet
- `final_metrics_comparison.png` - Bar chart comparing final validation metrics
- `convergence_analysis.png` - Convergence patterns with best epoch markers
- `overfitting_analysis.png` - Training vs validation gap analysis
- `training_time_comparison.png` - Training efficiency comparison
- `comparison_hyperparam_vs_xukuang.png` - Xukuang vs hyperparameter search comparison
- `comparison_why_xukuang_better.png` - Learning rate analysis

### Cross-Validation Study

**From validation_arch_comparison_20251013_093844/:**
- `arch_performance_comparison.png` - 5-fold CV results across architectures
- `arch_training_curves.png` - Detailed training dynamics per fold
- `arch_convergence_analysis.png` - Convergence analysis across folds

### Hyperparameter Optimization

**From hyperparameter_search_20251013_154754/:**
- `baseline_comparison.png` - All hyperparameter configurations vs baselines
- `hyperparam_effects_analysis.png` - Individual hyperparameter effect sizes
- `hyperparam_heatmaps.png` - Interaction effects between hyperparameters

### Density Analysis

**From density_analysis_dilution_factors/:**
- `density_by_dilution_mean_density.png` - Comprehensive comparison across all methods
- `density_clahe_otsu_only.png` - Reference method (CLAHE+OTSU) results
- `density_dl_models_only.png` - Deep learning models comparison

## Original Locations

Figures were copied from:
1. `../xukuang_params_shrunk_20251015_071224/`
2. `../validation_arch_comparison_20251013_093844/`
3. `../hyperparameter_search_20251013_154754/`
4. `../density_analysis_dilution_factors/`

## Usage

These figures are referenced in `COMPREHENSIVE_PROJECT_SUMMARY.md` using relative paths:
```markdown
![Description](figures/filename.png)
```

## Backup Strategy

To ensure figures persist:
1. All figures are copied (not symlinked) to this directory
2. This directory can be moved independently of source folders
3. If original experiment folders are archived/deleted, figures remain accessible

## Updates

If you need to update figures:
1. Re-run the corresponding experiment
2. Copy the new figure to this directory (overwrite)
3. Figures are automatically referenced in the main report

---

**Created:** October 25, 2025
**Purpose:** Preserve critical figures for project documentation
