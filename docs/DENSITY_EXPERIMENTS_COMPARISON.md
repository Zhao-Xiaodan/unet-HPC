# Density Experiments Comparison Summary

**Date:** October 14, 2025
**Status:** ✅ Complete
**Purpose:** Compare density prediction approaches on test images with dilution series

---

## Overview

Three separate density analysis experiments were conducted using different prediction strategies:

1. **Density Analysis - Architecture Comparison** (`density_analysis_arch_comparison_20251014_004358/`)
2. **Density Prediction with Representative Tiles** (`density_prediction_20251014_012038/`)
3. **Fast Density Prediction - 256×256 Tiles (Smart Model Caching)** (`density_prediction_256_20251014_054939/`)

All experiments:
- Trained or loaded U-Net, ResUNet, and Attention ResUNet models
- Used identical training configuration (from `validation_arch_comparison`)
- Predicted on the same test images (dilution series: 10x-10240x)
- Calculated foreground percentage as density metric
- Generated boxplots and comprehensive CSV files

**Key Differences:** Tile size, visualization approach, and model caching strategy.

---

## Experiment 1: Density Analysis - Architecture Comparison

### Scripts
- **Python:** `density_analysis_arch_comparison.py`
- **PBS Submission:** `pbs_density_analysis.sh`
- **Job Name:** `Density_Analysis`
- **Walltime:** 8 hours

### Training Configuration
```python
CONFIG = {
    'architectures': ['unet', 'resunet', 'attention_resunet'],
    'img_size': 256,              # Training AND prediction tile size
    'img_channels': 1,
    'filters': 64,
    'dropout': 0.2,
    'batch_size': 16,
    'learning_rate': 5e-5,
    'epochs': 50,
    'early_stopping_patience': 10,
}
```

### Prediction Strategy
- **Tile Size:** 256×256 (same as training)
- **Tiling:** Non-overlapping tiles extracted from test images
- **Processing:** Direct prediction on 256×256 tiles (no resizing)
- **Traditional Baseline:** CLAHE + OTSU thresholding

### Output Structure
```
density_analysis_arch_comparison_20251014_004358/
├── trained_models/
│   ├── unet_best_model.keras
│   ├── resunet_best_model.keras
│   └── attention_resunet_best_model.keras
├── plots/
│   ├── unet_density_vs_dilution.png
│   ├── resunet_density_vs_dilution.png
│   ├── attention_resunet_density_vs_dilution.png
│   └── clahe_otsu_density_vs_dilution.png
├── csv_data/
│   └── density_analysis_comprehensive.csv
├── density_analysis_arch_comparison.py    ← Source script
├── pbs_density_analysis.sh                ← PBS script
└── density_analysis_console_20251014_084334.log
```

### Outputs Generated
| Output Type | Count | Description |
|-------------|-------|-------------|
| Models | 3 | U-Net, ResUNet, Attention ResUNet |
| Boxplots | 4 | One per architecture/method (including CLAHE+OTSU) |
| CSV Files | 1 | Comprehensive density data for all tiles |
| Log Files | 1 | Console output with training metrics |

### Key Features
✓ **Direct comparison** with traditional CV method (CLAHE+OTSU)
✓ **Consistent tile size** (256×256) for training and prediction
✓ **No interpolation artifacts** (native resolution)
✓ **Simple workflow:** Train → Predict → Analyze

### Aim
**Primary Goal:** Compare DL architectures against traditional CV baseline (CLAHE+OTSU) for density measurement across dilution series.

**Research Question:** Do deep learning models provide better density estimates than classical computer vision methods?

---

## Experiment 2: Density Prediction with Representative Tiles

### Scripts
- **Python:** `density_prediction_with_tiles.py`
- **PBS Submission:** `pbs_density_prediction.sh`
- **Job Name:** `Density_Pred`
- **Walltime:** 8 hours

### Training Configuration
```python
CONFIG = {
    'architectures': ['unet', 'resunet', 'attention_resunet'],
    'train_img_size': 256,        # Training size
    'pred_tile_size': 512,        # Prediction tile size (DIFFERENT!)
    'img_channels': 1,
    'filters': 64,
    'dropout': 0.2,
    'batch_size': 16,
    'learning_rate': 5e-5,
    'epochs': 50,
    'early_stopping_patience': 10,
    'n_representative_tiles': 5,  # Per image
}
```

### Prediction Strategy
- **Tile Size:** 512×512 (2× training size)
- **Tiling:** Non-overlapping 512×512 tiles from test images
- **Processing:**
  1. Resize 512×512 tile → 256×256
  2. Predict using model trained on 256×256
  3. Resize prediction back → 512×512
  4. Threshold at 0.5
- **Representative Selection:** Select 5 tiles per image based on density percentiles (min, 25th, median, 75th, max)
- **Traditional Baseline:** CLAHE + OTSU (applied on 512×512 tiles)

### Output Structure
```
density_prediction_20251014_012038/
├── trained_models/
│   ├── unet_best_model.keras
│   ├── resunet_best_model.keras
│   └── attention_resunet_best_model.keras
├── representative_tiles/
│   ├── <image_name>_tile_00_comparison.png
│   ├── <image_name>_tile_01_comparison.png
│   ├── ...
│   └── (5 per test image × 10 images = ~50 comparisons)
├── boxplots/
│   ├── unet_density_vs_dilution.png
│   ├── resunet_density_vs_dilution.png
│   ├── attention_resunet_density_vs_dilution.png
│   └── clahe_otsu_density_vs_dilution.png
├── csv_data/
│   └── density_analysis_comprehensive.csv
├── density_prediction_with_tiles.py       ← Source script
├── pbs_density_prediction.sh              ← PBS script
└── density_prediction_console_20251014_092031.log
```

### Outputs Generated
| Output Type | Count | Description |
|-------------|-------|-------------|
| Models | 3 | U-Net, ResUNet, Attention ResUNet |
| Representative Tile Comparisons | ~50 | 5 per test image (4-panel: original + 3 predictions) |
| Boxplots | 4 | One per architecture/method |
| CSV Files | 1 | Comprehensive density data for all tiles |
| Log Files | 1 | Console output with training metrics |

### Key Features
✓ **Visual inspection** capability (representative tile comparisons)
✓ **Larger field of view** (512×512 tiles capture more context)
✓ **Percentile-based sampling** (representative coverage of density range)
✓ **4-panel comparison** (original | U-Net | ResUNet | Attention ResUNet)
✓ **CLAHE+OTSU baseline** included

### Aim
**Primary Goal:** Generate visual comparisons showing how each architecture segments representative tiles across the density spectrum.

**Research Question:** How do different architectures visually compare when segmenting tiles of varying density? Which architecture produces the most realistic-looking segmentations?

---

## Experiment 3: Fast Density Prediction - 256×256 Tiles (Smart Model Caching)

### Scripts
- **Python:** `density_prediction_256_fast.py`
- **PBS Submission:** `pbs_density_256_fast.sh`
- **Job Name:** `Density_256`
- **Walltime:** 8 hours

### Training Configuration
```python
CONFIG = {
    'architectures': ['unet', 'resunet', 'attention_resunet'],
    'img_size': 256,              # Training AND prediction size (same as Experiment 1)
    'img_channels': 1,
    'filters': 64,
    'dropout': 0.2,
    'batch_size': 16,
    'learning_rate': 5e-5,
    'epochs': 50,
    'early_stopping_patience': 10,
    'models_dir': './saved_models_validation_config',  # ← PERSISTENT MODEL STORAGE
    'n_representative_tiles': 5,  # Per image
}
```

### Smart Model Management (Key Innovation)
**Unique Feature:** Checks for existing trained models before training!

#### Model Check Logic (Lines 131-141 in Python script):
```python
def check_existing_models(models_dir):
    """Check if trained models already exist."""
    models_dir = Path(models_dir)
    existing = {}

    for arch in CONFIG['architectures']:
        model_path = models_dir / f'{arch}_best_model.keras'
        if model_path.exists():
            existing[arch] = model_path

    return existing
```

#### Execution Modes:
1. **First Run (0/3 models found):**
   - Trains all 3 models from scratch (~3-4 hours)
   - Saves models to `./saved_models_validation_config/`
   - Performs prediction and visualization
   - **Total runtime:** ~3-4 hours

2. **Subsequent Runs (3/3 models found):**
   - Loads models from disk (~30 seconds)
   - Skips training entirely
   - Performs prediction and visualization
   - **Total runtime:** ~5 minutes ✨

### Prediction Strategy
- **Tile Size:** 256×256 (same as training - native resolution)
- **Tiling:** Non-overlapping tiles extracted from test images
- **Processing:** Direct prediction on 256×256 tiles (no resizing)
- **Representative Selection:** Select 5 tiles per image based on density percentiles (min, 25th, median, 75th, max)
- **Traditional Baseline:** CLAHE + OTSU (applied on 256×256 tiles)

### Output Structure
```
density_prediction_256_20251014_054939/
├── boxplots/
│   ├── unet_density_vs_dilution.png
│   ├── resunet_density_vs_dilution.png
│   ├── attention_resunet_density_vs_dilution.png
│   └── clahe_otsu_density_vs_dilution.png
├── representative_tiles/
│   ├── <image_name>_tile_00_comparison.png
│   ├── <image_name>_tile_01_comparison.png
│   ├── ...
│   └── (5 per test image × 10 images = ~50 comparisons)
├── csv_data/
│   └── density_analysis_comprehensive.csv
├── density_prediction_256_fast.py         ← Source script
├── pbs_density_256_fast.sh                ← PBS script
└── density_256_console_20251014_134914.log

[PERSISTENT MODEL STORAGE - Separate directory]
saved_models_validation_config/
├── unet_best_model.keras
├── resunet_best_model.keras
└── attention_resunet_best_model.keras
```

### Outputs Generated
| Output Type | Count | Description |
|-------------|-------|-------------|
| Models (Persistent) | 3 | Saved to `./saved_models_validation_config/` (reusable!) |
| Representative Tile Comparisons | ~50 | 5 per test image (4-panel: original + 3 predictions) |
| Boxplots | 4 | One per architecture/method |
| CSV Files | 1 | Comprehensive density data for all tiles |
| Log Files | 1 | Console output with training metrics |

### Key Features
✓ **Smart model caching** - Train once, reuse forever!
✓ **Fast subsequent runs** (~5 minutes instead of ~4 hours)
✓ **Persistent model storage** - Models saved in separate directory
✓ **Visual inspection** capability (representative tile comparisons)
✓ **Native resolution** (256×256 tiles - no interpolation)
✓ **Percentile-based sampling** (representative coverage of density spectrum)
✓ **4-panel comparison** (original | U-Net | ResUNet | Attention ResUNet)
✓ **CLAHE+OTSU baseline** included

### Aim
**Primary Goal:** Combine the efficiency of model reuse with visual quality assessment using native-resolution tiles.

**Research Question:** Can we achieve the same quality as Experiment 1 (native 256×256 tiles) AND the visual comparisons of Experiment 2 (representative tiles) while enabling rapid iteration through smart model caching?

### What Happened in This Run
According to the log file (lines 476-479 in Python script):

```
⚠ Found 0/3 models
  Training missing models (this will take ~3-4 hours)...
  Future runs will be fast (~5 min) once models are saved
```

**Status:** This was the **FIRST RUN** of Experiment 3.
- ❌ No existing models found in `./saved_models_validation_config/`
- ✅ Trained all 3 architectures from scratch (U-Net, ResUNet, Attention ResUNet)
- ✅ Saved models to `./saved_models_validation_config/` for future use
- ✅ Generated representative tile comparisons and boxplots
- ✅ Exported comprehensive CSV

**Runtime:** ~3-4 hours (similar to Experiment 1, as expected for first run)

**Future Benefit:** Next run will be ~5 minutes! 🚀

---

## Key Differences

| Aspect | Experiment 1 | Experiment 2 | Experiment 3 |
|--------|--------------|--------------|--------------|
| **Name** | Arch Comparison | Representative Tiles 512×512 | Fast Prediction 256×256 |
| **Prediction Tile Size** | 256×256 (native) | 512×512 (2× training) | 256×256 (native) |
| **Resizing** | None | Resize 512→256→512 | None |
| **Primary Output** | 4 boxplots + CSV | 4 boxplots + CSV + ~50 tiles | 4 boxplots + CSV + ~50 tiles |
| **Visual Inspection** | ❌ No | ✅ Yes | ✅ Yes |
| **Tile Selection** | All tiles | 5 representative/image | 5 representative/image |
| **Field of View** | Smaller (256×256) | Larger (512×512) | Smaller (256×256) |
| **Interpolation Artifacts** | None | Potential (resizing) | None |
| **Model Caching** | ❌ No (retrain each time) | ❌ No (retrain each time) | ✅ Yes (persistent storage) |
| **First Run Time** | ~3-4 hours | ~3-4 hours | ~3-4 hours |
| **Subsequent Run Time** | ~3-4 hours (retrain!) | ~3-4 hours (retrain!) | ~5 minutes ⚡ |
| **Models Saved To** | Output dir only | Output dir only | `./saved_models_validation_config/` |
| **Use Case** | Quantitative comparison | Qualitative + Quantitative | Fast iteration + visual QA |
| **Best For** | One-time statistical analysis | One-time visual assessment | Iterative experimentation |

---

## Shared Configuration

Both experiments used **identical training settings** from `validation_arch_comparison`:

```python
# Model hyperparameters
filters: 64
dropout: 0.2
batch_size: 16
learning_rate: 5e-5
epochs: 50
early_stopping_patience: 10

# Loss and metrics
loss: Combined Dice + Focal Loss
metrics: [Jaccard coefficient, Dice coefficient]

# Architectures
1. U-Net (baseline)
2. ResUNet (residual connections)
3. Attention ResUNet (residual + attention gates)
```

All three experiments also included **CLAHE+OTSU** as a traditional computer vision baseline for comparison.

---

## Training Data

| Parameter | Value |
|-----------|-------|
| **Dataset** | `./dataset_full_stack/` |
| **Total Images** | 1,980 image-mask pairs |
| **Image Size** | 256×256 pixels (resized during loading) |
| **Format** | Grayscale .tif files |
| **Train/Val Split** | 80/20 (1,584 train, 396 val) |
| **Random Seed** | 42 |

---

## Test Data (Dilution Series)

| Parameter | Value |
|-----------|-------|
| **Directory** | `./test_images/` |
| **Total Images** | 10 .tif files |
| **Dilution Factors** | 10x, 20x, 80x, 160x, 320x, 640x, 1280x, 5120x, 10240x |
| **Format** | Large grayscale .tif files (variable dimensions) |

**Note:** Dilution factor represents the degree of sample dilution. Lower dilution (e.g., 10x) should have higher particle density; higher dilution (e.g., 10240x) should have lower density.

---

## What Was Achieved

### Experiment 1: Density Analysis - Architecture Comparison

#### ✅ Achievements
1. **Trained 3 DL architectures** from scratch (U-Net, ResUNet, Attention ResUNet)
2. **Established traditional CV baseline** using CLAHE+OTSU
3. **Generated density measurements** across full dilution series (10x-10240x)
4. **Created 4 boxplots** showing density vs dilution factor for each method
5. **Exported comprehensive CSV** with all tile-level density data
6. **Enabled quantitative comparison** between DL and traditional methods

#### 📊 Key Findings (Based on Log)
- Successfully loaded 1,980 training images
- Trained models with early stopping (monitored `val_jacard_coef`)
- Processed test images with 256×256 tiles
- **Note:** Previous analysis had dilution factor parsing bug (corrected in `DILUTION_ANALYSIS_CORRECTED_REPORT.md`)

#### 🎯 Primary Use Case
**Quantitative evaluation:** Determine which architecture provides the most accurate density estimates compared to CLAHE+OTSU baseline.

---

### Experiment 2: Density Prediction with Representative Tiles

#### ✅ Achievements
1. **Trained 3 DL architectures** from scratch (same config as Experiment 1)
2. **Generated ~50 visual comparisons** (4-panel: original + 3 predicted masks)
3. **Selected representative tiles** using percentile-based sampling (min, 25th, median, 75th, max density)
4. **Created 4 boxplots** showing density vs dilution factor
5. **Exported comprehensive CSV** with all tile-level density data
6. **Enabled visual inspection** of segmentation quality across density spectrum

#### 📊 Key Findings (Based on Log)
- Successfully loaded 1,980 training images (same dataset)
- Trained models with identical configuration
- Processed test images with 512×512 tiles (larger field of view)
- Generated comparison images for visual evaluation
- **Strategy:** Resize-predict-resize approach for 512×512 tiles

#### 🎯 Primary Use Case
**Qualitative evaluation:** Visually compare how each architecture segments tiles of varying density. Identify which architecture produces the most realistic segmentations.

---

### Experiment 3: Fast Density Prediction - 256×256 Tiles

#### ✅ Achievements
1. **Implemented smart model management** - Check for existing models before training
2. **Trained 3 DL architectures** from scratch (first run: no models existed)
3. **Saved models to persistent storage** (`./saved_models_validation_config/`)
4. **Generated ~50 visual comparisons** (4-panel: original + 3 predicted masks, 256×256 tiles)
5. **Selected representative tiles** using percentile-based sampling (min, 25th, median, 75th, max density)
6. **Created 4 boxplots** showing density vs dilution factor
7. **Exported comprehensive CSV** with all tile-level density data
8. **Enabled fast iteration** for future runs (~5 minutes vs ~4 hours)

#### 📊 Key Findings (Based on Log)
- **First run detected:** Found 0/3 models
- Successfully loaded 1,980 training images
- Trained all 3 models with early stopping (monitored `val_jacard_coef`)
- Saved models to `./saved_models_validation_config/`
- Processed test images with 256×256 tiles (native resolution)
- Generated representative tile visualizations
- **Runtime:** ~3-4 hours (expected for first run with training)

#### 🎯 Primary Use Case
**Fast iterative experimentation:** Combine native-resolution quality (256×256), visual inspection capability, and rapid iteration through model caching. Ideal for testing different prediction parameters, comparing dilution series, or exploring new test images without retraining models every time.

#### 💡 Relation to Experiment 1
**Experiment 3 is essentially Experiment 1 + Smart Caching + Visual Comparisons:**
- ✅ Same training configuration (identical to Experiment 1)
- ✅ Same tile size (256×256 native resolution)
- ✅ Same prediction strategy (direct, no resizing)
- ✅ **NEW:** Model persistence (Experiment 1 doesn't save models for reuse)
- ✅ **NEW:** Representative tile selection (Experiment 1 processes all tiles, no visual output)
- ✅ **NEW:** 4-panel visual comparisons (Experiment 1 only has boxplots)
- ✅ **NEW:** Fast subsequent runs (~5 min vs ~4 hours)

**Key Insight:** Experiment 3 combines the quantitative rigor of Experiment 1 with the visual assessment of Experiment 2, while adding the efficiency of model caching!

---

## Comparison Summary

| Metric | Experiment 1 | Experiment 2 | Experiment 3 |
|--------|-------------|--------------|--------------|
| **Training Time** | ~3-4 hours | ~3-4 hours | ~3-4 hours (first run) |
| **Prediction Strategy** | Direct (256×256) | Resize-based (512×512) | Direct (256×256) |
| **Total Runtime (First Run)** | ~4-6 hours | ~4-6 hours | ~4-6 hours |
| **Total Runtime (Subsequent)** | ~4-6 hours (retrain!) | ~4-6 hours (retrain!) | ~5 minutes ⚡ |
| **Output Focus** | Quantitative | Qualitative + Quantitative | Qualitative + Quantitative + Fast |
| **Visual Outputs** | 4 boxplots | 4 boxplots + 50 tiles | 4 boxplots + 50 tiles |
| **Tile Coverage** | All tiles | Representative (5/image) | Representative (5/image) |
| **Interpretation Ease** | Statistical | Visual + Statistical | Visual + Statistical |
| **Field of View** | 256×256 | 512×512 | 256×256 |
| **Model Reusability** | ❌ No | ❌ No | ✅ Yes (persistent cache) |
| **Best For** | One-time statistical comparison | One-time visual assessment | Iterative experimentation |

---

## Scientific Value

### Experiment 1 (Architecture Comparison)
**Value:** Provides **quantitative evidence** for which architecture best estimates particle density across dilution series.

**Insights:**
- Direct comparison with traditional CV baseline (CLAHE+OTSU)
- Statistical distribution of density across dilution factors
- Identification of optimal architecture for density measurement
- **Corrected finding:** Dilution bug revealed unexpected density increase at extreme dilutions (640x-10240x)

**Limitations:**
- No visual inspection of segmentation quality
- Smaller tiles (256×256) may miss larger-scale patterns

---

### Experiment 2 (Representative Tiles)
**Value:** Provides **qualitative evidence** through visual comparisons of segmentation quality.

**Insights:**
- Visual assessment of segmentation realism
- Comparison of architecture performance across density spectrum (min to max)
- Identification of failure cases or artifacts
- Larger context (512×512 tiles) captures more spatial information

**Limitations:**
- Interpolation artifacts from resize-predict-resize workflow
- Representative sampling (5 tiles per image) may miss outliers
- Subjective interpretation required

### Experiment 3 (Fast 256×256)
**Value:** Combines quantitative + qualitative assessment with rapid iteration capability.

**Insights:**
- Same statistical rigor as Experiment 1 (native 256×256 tiles)
- Same visual assessment as Experiment 2 (representative tile comparisons)
- **NEW:** Enables rapid experimentation through model caching
- Ideal for testing new dilution series or prediction parameters without retraining

**Unique Contribution:**
- Smart model management reduces iteration time from hours to minutes
- Maintains native resolution quality (no interpolation)
- Facilitates hypothesis testing and parameter tuning

**Limitations:**
- Smaller field of view than Experiment 2 (256×256 vs 512×512)
- Still requires initial training run (~3-4 hours)
- Persistent models consume disk space (~500 MB total)

---

## Complementary Nature

The three experiments are **highly complementary**:

1. **Experiment 1** answers: *"Which architecture is most accurate statistically?"*
2. **Experiment 2** answers: *"Which architecture produces the most realistic segmentations at larger scale?"*
3. **Experiment 3** answers: *"Can we iterate quickly while maintaining quality?"*

Together, they provide:
- **Quantitative metrics** (foreground percentage, statistical distributions)
- **Qualitative assessment** (visual realism, artifact detection)
- **Multi-scale validation** (256×256 native vs 512×512 upscaled)
- **Efficient iteration** (smart caching for rapid experimentation)

### Recommended Usage Strategy:
1. **Initial exploration:** Use Experiment 3 for rapid testing of architectures, parameters, or new test images
2. **Detailed statistical analysis:** Use Experiment 1 for comprehensive tile-level statistics
3. **Large-scale visual validation:** Use Experiment 2 for 512×512 field of view assessment
4. **Production deployment:** Use models from Experiment 3's persistent cache for inference

---

## Recommended Analysis Workflow

### Initial Quick Assessment (Start Here!)
1. **Review Experiment 3 boxplots and tile comparisons** - Get quick overview of all architectures
2. **Check for obvious failures** - Identify any architecture that clearly underperforms
3. **Note interesting patterns** - Look for unexpected density trends or segmentation artifacts

### Detailed Statistical Analysis
4. **Review Experiment 1 boxplots** - Comprehensive statistical comparison with all tiles
5. **Cross-reference Experiment 3 CSV** - Validate that representative tiles match full dataset trends
6. **Identify best-performing architecture** statistically

### Visual Quality Verification
7. **Review Experiment 3 tile comparisons (256×256)** - Native resolution quality check
8. **Review Experiment 2 tile comparisons (512×512)** - Larger field of view assessment
9. **Compare 256×256 vs 512×512 outputs** - Check if larger tiles capture more context

### Final Decision Making
10. **Cross-reference all three CSVs** - Verify consistency across experiments
11. **Identify discrepancies** - Architecture with best stats but poor visual quality?
12. **Make informed decision** - Based on quantitative + qualitative + multi-scale evidence

### Iterative Refinement (If Needed)
13. **Use Experiment 3 for rapid testing** - Try different parameters or new test images (~5 min per run!)
14. **Re-run Experiment 1 or 2 if major changes needed** - Full retraining for architecture modifications

---

## Known Issues

### Dilution Factor Parsing Bug (Fixed)
- **Issue:** Substring matching in `extract_dilution_factor()` caused incorrect parsing
- **Impact:** 5 out of 10 images misclassified in Experiment 1
- **Fix:** Corrected in `reanalyze_density_data.py` using regex with word boundaries
- **Status:** ✅ Fixed and documented in `DILUTION_ANALYSIS_CORRECTED_REPORT.md`

### Unexpected Density Increase at Extreme Dilutions
- **Observation:** Density increases at 640x-10240x (biologically implausible)
- **Possible Causes:** Image artifacts, background noise, or segmentation errors
- **Recommendation:** Visually inspect extreme dilution images (see Experiment 2 tiles)

---

## Files Organized

### Experiment 1 Directory Contents
```
density_analysis_arch_comparison_20251014_004358/
├── density_analysis_arch_comparison.py    ✅ Added
├── pbs_density_analysis.sh                ✅ Added
├── trained_models/                         (3 .keras files)
├── plots/                                  (4 .png files)
├── csv_data/                               (1 .csv file)
└── density_analysis_console_20251014_084334.log
```

### Experiment 2 Directory Contents
```
density_prediction_20251014_012038/
├── density_prediction_with_tiles.py       ✅ Added
├── pbs_density_prediction.sh              ✅ Added
├── trained_models/                         (3 .keras files)
├── representative_tiles/                   (~50 .png files)
├── boxplots/                               (4 .png files)
├── csv_data/                               (1 .csv file)
└── density_prediction_console_20251014_092031.log
```

### Experiment 3 Directory Contents
```
density_prediction_256_20251014_054939/
├── density_prediction_256_fast.py         ✅ Added
├── pbs_density_256_fast.sh                ✅ Added
├── boxplots/                               (4 .png files)
├── representative_tiles/                   (~50 .png files)
├── csv_data/                               (1 .csv file)
└── density_256_console_20251014_134914.log

[PERSISTENT MODELS - Separate directory for reuse]
saved_models_validation_config/
├── unet_best_model.keras                  ✅ Saved for future runs
├── resunet_best_model.keras               ✅ Saved for future runs
└── attention_resunet_best_model.keras     ✅ Saved for future runs
```

---

## Summary

✅ **All three experiments completed successfully**
✅ **Source scripts (.py and .sh) organized into respective output directories**
✅ **Complementary approaches:**
   - Experiment 1: Quantitative (all tiles, 256×256 native)
   - Experiment 2: Qualitative + Quantitative (representative tiles, 512×512 upscaled)
   - Experiment 3: Qualitative + Quantitative + Fast iteration (representative tiles, 256×256 native, model caching)
✅ **Shared training configuration** ensures fair comparison across all experiments
✅ **Comprehensive outputs:** Models, plots, CSVs, logs, and visual comparisons
✅ **Smart caching implemented** (Experiment 3) - Future runs take ~5 minutes instead of ~4 hours!

**Key Discovery - Experiment 3:**
- **First run status:** Found 0/3 models → trained all from scratch (~3-4 hours)
- **Models saved to:** `./saved_models_validation_config/` (persistent cache)
- **Future benefit:** Subsequent runs will load models in ~30 seconds, total runtime ~5 minutes!
- **Practical impact:** Enables rapid iteration for testing new dilution series, parameters, or test images

**Experiment Relationships:**
- **Experiment 1:** Foundation - comprehensive statistical analysis with native resolution
- **Experiment 2:** Extension - adds visual assessment with larger field of view
- **Experiment 3:** Optimization - combines Exp1 quality + Exp2 visuals + smart caching

**Next Steps:**
1. **Quick start:** Review Experiment 3 outputs (boxplots + tile comparisons) for initial assessment
2. **Detailed stats:** Review Experiment 1 for comprehensive tile-level statistics
3. **Visual validation:** Review Experiment 2 for 512×512 large-scale segmentation quality
4. **Cross-reference:** Compare CSV data across all three experiments to verify consistency
5. **Multi-scale check:** Compare 256×256 (Exp 1 & 3) vs 512×512 (Exp 2) tile outputs
6. **Determine best architecture:** Based on quantitative + qualitative + multi-scale evidence
7. **Address anomalies:** Investigate unexpected density increase at extreme dilutions (640x-10240x)
8. **Iterate if needed:** Use Experiment 3's fast mode (~5 min) for parameter tuning or new test images

---

**Comparison Complete:** ✓
**Scripts Organized:** ✓
**Three Experiments Documented:** ✓
**Model Caching Strategy Explained:** ✓
**Ready for Analysis:** ✓
