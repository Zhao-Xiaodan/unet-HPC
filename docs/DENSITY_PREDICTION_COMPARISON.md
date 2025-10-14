# Density Prediction: Three Approaches Compared

**Created:** October 14, 2025
**Purpose:** Compare three density prediction workflows and help you choose the right one

---

## Overview: Which Script Should You Use?

| Script | Use Case | Runtime | Training? |
|--------|----------|---------|-----------|
| **density_prediction_existing_models.py** | ✅ **RECOMMENDED** - Fast prediction using existing models | **~5 min** | ❌ No |
| **density_prediction_256_fast.py** | Smart caching - trains once, fast thereafter | First: 3-4 hr, Later: 5 min | ⚠️ If needed |
| **density_prediction_with_tiles.py** | 512×512 tiles (for comparison with 256×256) | **~4-6 hr** | ✅ Always |

---

## Approach 1: Existing Models (RECOMMENDED)

### Files:
- `density_prediction_existing_models.py`
- `pbs_density_existing_models.sh`

### What It Does:
```
1. Search for trained models in validation_arch_comparison_20251013_093844
2. Read results.json files to find best fold per architecture
3. Load best models (U-Net, ResUNet, Attention ResUNet)
4. Predict on test images with 256×256 tiles
5. Generate representative tiles + boxplots + CSV
```

### Key Features:
✅ **No training** - uses existing models
✅ **Fast execution** - ~5 minutes
✅ **Native 256×256 resolution** - no interpolation
✅ **Best fold selection** - automatically finds best performing fold

### How It Works:

#### Model Search Strategy:
```python
def search_for_model(architecture):
    """Search multiple locations and fold subdirectories."""
    search_paths = [
        './validation_arch_comparison_20251013_093844',
        './microscope_training_20251008_074915',
        './saved_models_validation_config',
    ]

    for path in search_paths:
        # Try direct model files
        for pattern in [f'{arch}_best_model.keras', f'best_{arch}.h5']:
            if model_file.exists():
                return model_file

        # Try fold subdirectories
        arch_dir = path / architecture
        if arch_dir.exists():
            best_fold, best_jaccard = find_best_fold_from_results(arch_dir)
            if best_fold is not None:
                # Load model from best fold
                fold_dir = arch_dir / f'fold_{best_fold}'
                return find_model_in_folder(fold_dir)
```

#### Best Fold Selection:
```python
def find_best_fold_from_results(arch_dir):
    """Read results.json files to find best fold."""
    best_fold = None
    best_jaccard = -1

    for fold_dir in arch_dir.glob('fold_*'):
        results_file = fold_dir / 'results.json'
        if results_file.exists():
            results = json.load(open(results_file))
            jaccard = results.get('best_val_jacard', -1)

            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_fold = int(fold_dir.name.split('_')[1])

    return best_fold, best_jaccard
```

### Expected Directory Structure:
```
validation_arch_comparison_20251013_093844/
├── unet/
│   ├── fold_1/
│   │   ├── best_model.keras  ← Model file
│   │   ├── results.json      ← Contains best_val_jacard
│   │   └── history.csv
│   ├── fold_2/
│   │   └── ...
│   └── fold_3/
│       └── ...
├── resunet/
│   ├── fold_1/
│   └── ...
└── attention_resunet/
    ├── fold_1/
    └── ...
```

### Runtime Breakdown:
```
Total: ~5 minutes

  Search for models:          ~5 sec
  Find best fold (3 archs):   ~5 sec
  Load 3 models:             ~20 sec
  Predict on test images:    ~3-4 min
  Generate outputs:          ~1 min
```

### How to Run:
```bash
# Transfer to HPC
scp density_prediction_existing_models.py \
    pbs_density_existing_models.sh \
    phyzxi@atlas7.nus.edu.sg:~/scratch/unet-HPC/

# Submit job
ssh phyzxi@atlas7.nus.edu.sg
cd ~/scratch/unet-HPC
qsub pbs_density_existing_models.sh

# Monitor
qstat -u phyzxi
tail -f Density_Exist.o*  # Once running
```

### When to Use:
✅ You have trained models from validation_arch_comparison
✅ You want fast predictions (~5 min)
✅ You need to re-run predictions multiple times
✅ You want native 256×256 resolution (best for small particles)

---

## Approach 2: Smart Caching (256×256 Native)

### Files:
- `density_prediction_256_fast.py`
- `pbs_density_256_fast.sh`

### What It Does:
```
First run:
1. Check for existing models in ./saved_models_validation_config/
2. If not found: Train models, save them
3. Predict on test images
4. Generate outputs

Subsequent runs:
1. Load saved models (fast!)
2. Predict on test images
3. Generate outputs
```

### Key Features:
✅ **Trains once, reuses forever**
✅ **256×256 native resolution**
✅ **Self-contained** - manages its own models
⚠️ Trains from scratch on first run (uses validation_arch_comparison config)

### Runtime:
- **First run:** ~3-4 hours (trains + saves models)
- **Subsequent runs:** ~5 minutes (loads saved models)

### How It Works:
```python
def check_existing_models(models_dir):
    """Check if models already exist."""
    existing = {}
    for arch in ['unet', 'resunet', 'attention_resunet']:
        model_path = models_dir / f'{arch}_best_model.keras'
        if model_path.exists():
            existing[arch] = model_path
    return existing

# In main():
existing_models = check_existing_models('./saved_models_validation_config')

if len(existing_models) == 3:
    print("✓ All models found - FAST MODE")
    for arch in CONFIG['architectures']:
        model = load_model(existing_models[arch])
else:
    print("⚠ Training models (first run)")
    for arch in CONFIG['architectures']:
        model = train_model(arch, X_train, y_train, X_val, y_val)
        save_model(model, f'{arch}_best_model.keras')
```

### Saved Models Location:
```
saved_models_validation_config/
├── unet_best_model.keras           (~100 MB)
├── resunet_best_model.keras        (~100 MB)
└── attention_resunet_best_model.keras  (~100 MB)

Total: ~300 MB (persistent across runs)
```

### When to Use:
✅ You don't have existing models yet
✅ You're willing to wait 3-4 hours for first run
✅ You'll run predictions many times in the future
✅ You want self-contained script with no external dependencies

---

## Approach 3: 512×512 Tiles (For Comparison)

### Files:
- `density_prediction_with_tiles.py`
- `pbs_density_prediction.sh`

### What It Does:
```
1. Train models (always, every run)
2. Extract 512×512 tiles from test images
3. Resize to 256×256 for prediction (model trained on 256×256)
4. Resize prediction back to 512×512
5. Generate outputs
```

### Key Features:
⚠️ **Always trains** - no model reuse
⚠️ **Resolution mismatch** - downsampling loses detail
✅ **Larger context** - 512×512 tiles
✅ **Useful for comparison** - validate that 256×256 is better

### Runtime:
- **Every run:** ~4-6 hours (always trains)

### Why Use 512×512?

**For Comparison Purposes:**
Your observation: "dozens of particles in one cropped image" at 256×256 suggests that 256×256 provides sufficient context. Running 512×512 validates this hypothesis by showing:

1. **Resolution loss:** Downsampling 512→256 makes particles appear half size
2. **Interpolation artifacts:** Upsampling 256→512 introduces blur
3. **Statistical differences:** Fewer tiles (16 vs 64 per image)

### Resize Strategy:
```python
def predict_on_tile_with_model(model, tile_512):
    # Original 512×512 tile
    # Particle diameter: ~30 pixels

    # Step 1: Downsample to 256×256
    tile_256 = cv2.resize(tile_512, (256, 256))
    # Particle diameter: ~15 pixels (50% size)

    # Step 2: Predict using model
    pred_256 = model.predict(tile_256)

    # Step 3: Upsample back to 512×512
    pred_512 = cv2.resize(pred_256, (512, 512))
    # Result: Blurry boundaries due to interpolation

    return pred_512
```

### When to Use:
✅ You want to compare 512×512 vs 256×256 results
✅ You need to validate that native resolution is better
✅ You're investigating whether larger context helps
⚠️ Not recommended for production use (slow, less accurate)

---

## Side-by-Side Comparison

### Feature Matrix:

| Feature | Existing Models | Smart Caching | 512×512 Tiles |
|---------|----------------|---------------|---------------|
| **First run time** | 5 min | 3-4 hours | 4-6 hours |
| **Repeat run time** | 5 min | 5 min | 4-6 hours |
| **Training phase** | ❌ None | ⚠️ First run only | ✅ Every run |
| **Tile size** | 256×256 | 256×256 | 512×512 |
| **Prediction resolution** | Native | Native | Downsampled |
| **Interpolation artifacts** | ❌ None | ❌ None | ⚠️ Yes |
| **Tiles per 2048×2048 image** | 64 | 64 | 16 |
| **Statistical robustness** | High (more samples) | High (more samples) | Lower (fewer samples) |
| **Model source** | validation_arch_comparison | Trains new | Trains new |
| **Best fold selection** | ✅ Automatic | ❌ N/A | ❌ N/A |
| **Disk usage (models)** | 0 MB (uses existing) | ~300 MB | ~300 MB per run |

### Output Comparison:

All three approaches produce:

```
density_prediction_*/
├── representative_tiles/
│   └── [image]_tile_[idx]_comparison.png  (4-panel: original + 3 masks)
├── boxplots/
│   ├── unet_density_vs_dilution.png
│   ├── resunet_density_vs_dilution.png
│   ├── attention_resunet_density_vs_dilution.png
│   └── clahe_otsu_density_vs_dilution.png
└── csv_data/
    └── density_analysis_comprehensive.csv
```

**Only difference:**
- **Existing Models & Smart Caching:** No trained_models/ subdirectory in output
- **512×512 Tiles:** Includes trained_models/ with 3 .keras files

---

## Recommended Workflow

### Step 1: Use Existing Models (Fast Prediction)
```bash
qsub pbs_density_existing_models.sh
```
**Why:** Fastest way to get results using models you already trained

### Step 2: Compare with 512×512 (Validation)
```bash
qsub pbs_density_prediction.sh
```
**Why:** Validate that native 256×256 resolution is indeed better

### Step 3: Analyze Results
```python
import pandas as pd

# Load both CSVs
df_256 = pd.read_csv('density_prediction_existing_*/csv_data/density_analysis_comprehensive.csv')
df_512 = pd.read_csv('density_prediction_*/csv_data/density_analysis_comprehensive.csv')

# Compare mean density
df_256.groupby('method')['foreground_pct'].mean()
df_512.groupby('method')['foreground_pct'].mean()

# Compare variance (256 should be lower due to more samples)
df_256.groupby('method')['foreground_pct'].std()
df_512.groupby('method')['foreground_pct'].std()

# Compare sample counts
len(df_256)  # 64 tiles × 11 images × 4 methods = 2816 measurements
len(df_512)  # 16 tiles × 11 images × 4 methods = 704 measurements
```

### Step 4: Visual Comparison
```bash
# Compare tile quality side-by-side
# 256×256: sharp boundaries, small particles detected
# 512×512: blurry boundaries, interpolation artifacts
```

---

## Technical Details

### Representative Tile Selection

All three approaches use the same method:

1. Predict on ALL tiles using U-Net
2. Sort tiles by foreground percentage (density)
3. Select 5 tiles representing distribution:

```python
tile_densities = sorted(unet_densities)  # [0.1%, 0.5%, 1.2%, 2.3%, 5.7%, ...]

selected_indices = [
    0,                          # Minimum (0.1%)
    len(tiles) // 4,            # 25th percentile (1.0%)
    len(tiles) // 2,            # Median (2.3%)
    3 * len(tiles) // 4,        # 75th percentile (4.2%)
    len(tiles) - 1,             # Maximum (5.7%)
]
```

This ensures representative coverage: sparse regions, typical regions, and dense regions.

### 4-Panel Comparison Visualization

Each representative tile generates one PNG:

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│   Original   │    U-Net     │   ResUNet    │  Attention   │
│   Grayscale  │  Prediction  │  Prediction  │   ResUNet    │
│              │              │              │  Prediction  │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ [Raw image]  │ [Mask]       │ [Mask]       │ [Mask]       │
│              │ 2.34%        │ 2.12%        │ 2.45%        │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

**Key feature:** Same tile predicted by all 3 architectures for direct comparison.

### Density Calculation

```python
def calculate_foreground_percentage(binary_mask):
    """
    binary_mask: 0 (background) or 255 (foreground)
    Returns: foreground percentage (0-100%)
    """
    foreground_pixels = np.sum(binary_mask == 255)
    total_pixels = binary_mask.size
    foreground_pct = (foreground_pixels / total_pixels) * 100
    return foreground_pct
```

### Boxplot Configuration

All three approaches generate 4 separate boxplots:

```python
plt.figure(figsize=(12, 8))

# Y-axis: log scale
plt.yscale('log')
plt.ylabel('Foreground Percentage', fontsize=14)

# X-axis: 1/Dilution Factor
x_labels = [f'1/{d}' for d in sorted(dilution_factors)]
plt.xlabel('1/Dilution Factor', fontsize=14)

# One boxplot per method
for method in ['unet', 'resunet', 'attention_resunet', 'clahe_otsu']:
    data_by_dilution = [df[df['dilution_factor'] == d]['foreground_pct']
                        for d in sorted(dilution_factors)]
    plt.boxplot(data_by_dilution, positions=range(len(dilution_factors)))
    plt.savefig(f'{method}_density_vs_dilution.png', dpi=300, bbox_inches='tight')
```

---

## Troubleshooting

### Issue 1: Models Not Found (Existing Models Approach)

**Error:**
```
ERROR: Could not find model for unet
Searched in: validation_arch_comparison_20251013_093844
```

**Diagnosis:**
```bash
# Check if model files exist
ls -la validation_arch_comparison_20251013_093844/unet/fold_1/

# Look for:
#   - *.keras files
#   - *.h5 or *.hdf5 files
#   - results.json (for best fold selection)
```

**Solution:**
1. If models exist but not detected: Check file naming patterns in script
2. If models don't exist: Use Smart Caching approach instead (trains and saves)

### Issue 2: Best Fold Selection Fails

**Error:**
```
WARNING: Could not determine best fold for unet from results.json
```

**Cause:** results.json files missing or malformed

**Solution:**
```python
# Manually specify fold in script:
CONFIG['manual_fold_selection'] = {
    'unet': 1,          # Use fold_1
    'resunet': 2,       # Use fold_2
    'attention_resunet': 1,
}
```

### Issue 3: Memory Error During Prediction

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solution:**
Reduce batch size in script:
```python
CONFIG['batch_size'] = 8  # Default: 16, try 8 or 4
```

### Issue 4: 512×512 vs 256×256 Results Too Similar

**Question:** "Why do 512×512 and 256×256 give similar densities?"

**Answer:** Mean densities may be similar, but check:
1. **Visual quality:** 256×256 masks should be sharper
2. **Variance:** 256×256 should have lower std deviation (more samples)
3. **Small particles:** 256×256 should detect more small particles

**Validation:**
```python
# Compare coefficient of variation (CV)
cv_256 = df_256.groupby('method')['foreground_pct'].std() / df_256.groupby('method')['foreground_pct'].mean()
cv_512 = df_512.groupby('method')['foreground_pct'].std() / df_512.groupby('method')['foreground_pct'].mean()

# 256×256 should have lower CV (more stable estimates)
print(f'CV 256: {cv_256}')
print(f'CV 512: {cv_512}')
```

---

## Summary: Which Approach to Choose?

### ✅ **Use Existing Models** if:
- You have trained models from validation_arch_comparison ✅
- You want fastest predictions (~5 min) ✅
- You'll run predictions multiple times ✅
- You want native 256×256 resolution ✅

### ⚠️ **Use Smart Caching** if:
- You don't have existing models yet
- You're willing to wait 3-4 hours for first run
- You want self-contained script
- Future runs need to be fast

### 📊 **Use 512×512 Tiles** if:
- You want to validate 256×256 is better
- You're investigating resolution effects
- You need to justify native resolution choice
- You're okay with slower execution

---

## Files Summary

### Existing Models (Recommended):
✅ `density_prediction_existing_models.py` (17 KB)
✅ `pbs_density_existing_models.sh` (8 KB, executable)

### Smart Caching (256×256):
✅ `density_prediction_256_fast.py` (17 KB)
✅ `pbs_density_256_fast.sh` (8 KB, executable)

### 512×512 Tiles (Comparison):
✅ `density_prediction_with_tiles.py` (14 KB)
✅ `pbs_density_prediction.sh` (8 KB, executable)

### Documentation:
✅ `COMPARISON_512_vs_256.md` (12 KB) - Detailed 512 vs 256 analysis
✅ `DENSITY_PREDICTION_GUIDE.md` (12 KB) - Original guide
✅ `DENSITY_PREDICTION_COMPARISON.md` (this file) - Complete comparison

---

**Ready to use! Transfer files to HPC and submit the appropriate PBS script.**
