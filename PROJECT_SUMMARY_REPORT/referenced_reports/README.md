# Referenced Reports

This directory contains copies of all markdown reports referenced in the comprehensive project summary. All figures in these reports have been updated to use relative paths pointing to `../figures/`.

## Purpose

These reports provide detailed documentation of each experimental phase. They have been copied here to ensure the PROJECT_SUMMARY_REPORT remains self-contained and portable, even if the original experiment folders are moved or deleted.

## Contents

### Phase 1 & 2: Initial Failures

**[PHASE1_RESULTS_ANALYSIS.md](PHASE1_RESULTS_ANALYSIS.md)**
- First validation test with microbead dataset
- Result: 13.8% → 3% IoU (catastrophic collapse)
- Original location: `[99]Archive/microbeads/`

**[SMALL_MODEL_RESULTS_ANALYSIS.md](SMALL_MODEL_RESULTS_ANALYSIS.md)**
- Test of smaller model (16 filters, 2M params)
- Result: 7.6% IoU (worse than large model!)
- Original location: `[99]Archive/microbeads/`

**[FOCAL_TVERSKY_TEST_RESULTS.md](FOCAL_TVERSKY_TEST_RESULTS.md)**
- Test of Focal Tversky loss function
- Result: 13.3% → 2.1% IoU (even worse)
- Original location: `[99]Archive/microbeads/`

### Phase 3: Root Cause Discovery

**[CRITICAL_TRAINING_FAILURE_ANALYSIS.md](CRITICAL_TRAINING_FAILURE_ANALYSIS.md)**
- **Critical discovery:** FP16 mixed precision causing NaN/inf in losses
- Evidence: Training logs showing NaN values
- Solution: Switch to FP32 precision
- Original location: `docs/`

### Phase 4: Success with Xukuang's Parameters

**[README_XUKUANG_SHRUNK.md](README_XUKUANG_SHRUNK.md)**
- Guide to Xukuang's parameters from bead_seg.ipynb
- Parameters: LR=5e-3, 200 epochs, BinaryFocalLoss
- Original location: Project root

**[XUKUANG_PARAMS_REPORT.md](XUKUANG_PARAMS_REPORT.md)**
- Detailed analysis of training with Xukuang's parameters
- Best result: UNet 67.9% IoU (epoch 140)
- Attention mechanisms analysis (unstable)
- Original location: `xukuang_params_shrunk_20251015_071224/report.md`

### Phase 5: Cross-Validation

**[VALIDATION_ARCH_COMPARISON_REPORT.md](VALIDATION_ARCH_COMPARISON_REPORT.md)**
- 5-fold cross-validation across 3 architectures
- UNet: 69.94% ± 5.02% (best)
- ResUNet catastrophic failure analysis
- Original location: `validation_arch_comparison_20251013_093844/REPORT.md`

### Phase 6: Hyperparameter Optimization

**[HYPERPARAMETER_SEARCH_REPORT.md](HYPERPARAMETER_SEARCH_REPORT.md)**
- Grid search: 19 configs × 3 folds = 57 runs
- Best: ResUNet lr5e-05_drop0.3_bs8 (60.05% IoU)
- Learning rate effects: 5e-5 >> 2e-5 > 1e-5
- Original location: `hyperparameter_search_20251013_154754/REPORT.md`

### Phase 7: PyTorch Implementation

**[PYTORCH_COMPARISON_RESULTS.md](PYTORCH_COMPARISON_RESULTS.md)** ⭐ **PRIMARY REPORT**
- **Comprehensive analysis of 243 PyTorch models**
- 3 experiments × 3 architectures × 27 hyperparameter configs
- Key findings:
  - Augmentation hurt performance (-4.4% for UNet)
  - AdaptiveLoss unstable (10 failures vs 0 for BinaryFocal)
  - UNet best: 64.17% IoU
- Original location: Project root (PYTORCH_COMPARISON_RESULTS.md)

**[PYTORCH_EXPERIMENTS_COMPARISON.md](PYTORCH_EXPERIMENTS_COMPARISON.md)**
- Earlier summary of PyTorch experiments
- Identified that best models come from different experiments
- Original location: `share_folder/PYTORCH_EXPERIMENTS_COMPARISON.md`

### Phase 8: Density Analysis

**[DENSITY_ANALYSIS_REPORT.md](DENSITY_ANALYSIS_REPORT.md)**
- Validation using dilution series (10× to 10240×)
- CLAHE+OTSU reference method (11.99% to 64.80% density)
- Deep learning model validation
- 440 tile measurements across 11 images
- Original location: `density_analysis_dilution_factors/`

## Figure Paths

All figures in these reports use relative paths to `../figures/`:

**Example:**
```markdown
![Figure](../figures/training_curves_comparison.png)
```

This ensures figures remain accessible when the PROJECT_SUMMARY_REPORT directory is moved or shared independently.

## Usage

### For Reading Reports
Open any .md file in a markdown viewer. All figure links will resolve correctly as long as the `figures/` directory exists one level up.

### For Citation
When citing these reports in presentations or papers, refer to both the report and its original experiment folder:

**Example:**
> "As documented in the Xukuang Parameters Report (xukuang_params_shrunk_20251015_071224), the UNet model achieved 67.9% validation IoU at epoch 140..."

## Modification History

All reports in this directory are copies from specific experiment folders. They have been modified minimally:
- ✅ Figure paths updated to `../figures/`
- ✅ File names standardized (some renamed for clarity)
- ❌ Content NOT modified (preserved as written)

## Original Locations Reference

| Report File | Original Location |
|-------------|-------------------|
| PHASE1_RESULTS_ANALYSIS.md | [99]Archive/microbeads/PHASE1_RESULTS_ANALYSIS.md |
| SMALL_MODEL_RESULTS_ANALYSIS.md | [99]Archive/microbeads/SMALL_MODEL_RESULTS_ANALYSIS.md |
| FOCAL_TVERSKY_TEST_RESULTS.md | [99]Archive/microbeads/FOCAL_TVERSKY_TEST_RESULTS.md |
| CRITICAL_TRAINING_FAILURE_ANALYSIS.md | docs/CRITICAL_TRAINING_FAILURE_ANALYSIS.md |
| README_XUKUANG_SHRUNK.md | README_XUKUANG_SHRUNK.md |
| XUKUANG_PARAMS_REPORT.md | xukuang_params_shrunk_20251015_071224/report.md |
| VALIDATION_ARCH_COMPARISON_REPORT.md | validation_arch_comparison_20251013_093844/REPORT.md |
| HYPERPARAMETER_SEARCH_REPORT.md | hyperparameter_search_20251013_154754/REPORT.md |
| PYTORCH_COMPARISON_RESULTS.md | PYTORCH_COMPARISON_RESULTS.md (root) |
| PYTORCH_EXPERIMENTS_COMPARISON.md | share_folder/PYTORCH_EXPERIMENTS_COMPARISON.md |
| DENSITY_ANALYSIS_REPORT.md | density_analysis_dilution_factors/DENSITY_ANALYSIS_REPORT.md |

---

**Total Reports:** 11
**Total Size:** ~500 KB
**Created:** October 25, 2025
**Purpose:** Preserve complete project documentation for long-term reference
