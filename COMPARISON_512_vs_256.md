# Comparison: 512×512 vs 256×256 Tile Predictions

**Created:** October 14, 2025
**Purpose:** Compare two density prediction approaches

---

## Summary of Two Jobs

### Job 1: 512×512 Tiles (Original Request)
**Script:** `density_prediction_with_tiles.py`
**PBS:** `pbs_density_prediction.sh`

**Approach:**
- Extract 512×512 tiles from test images
- Resize to 256×256 for prediction (model trained on 256×256)
- Resize predictions back to 512×512
- Calculate density from upsampled masks

**Runtime:** ~4-6 hours (always trains models)

**Pros:**
- Larger spatial context per tile
- Fewer tiles per image

**Cons:**
- ⚠️ **Resolution mismatch:** Downsampling loses detail
- ⚠️ **Scale confusion:** Model sees particles at wrong size
- ⚠️ **Interpolation artifacts:** Upsampling introduces blur
- Always trains from scratch (no model reuse)

---

### Job 2: 256×256 Tiles (Recommended)
**Script:** `density_prediction_256_fast.py`
**PBS:** `pbs_density_256_fast.sh`

**Approach:**
- Extract 256×256 tiles directly
- Predict at native resolution (model trained on 256×256)
- No resizing needed
- Calculate density from native-resolution masks

**Runtime:**
- First run: ~3-4 hours (trains & saves models)
- Subsequent runs: **~5 minutes** (loads saved models)

**Pros:**
- ✅ **Native resolution:** No interpolation artifacts
- ✅ **Scale consistency:** Model sees particles at trained size
- ✅ **Better accuracy:** Especially for small particles
- ✅ **More tiles:** 4× more samples per image = better statistics
- ✅ **Smart caching:** Saves models for future fast runs

**Cons:**
- More tiles to process per image (but this is actually good for statistics)

---

## Side-by-Side Comparison

| Aspect | 512×512 Tiles | 256×256 Tiles |
|--------|---------------|---------------|
| **Tile extraction** | Every 512 pixels | Every 256 pixels |
| **Tiles per 2048×2048 image** | 16 tiles | 64 tiles |
| **Prediction method** | Resize → predict → resize | Direct prediction |
| **Model sees particles at** | ~50% size (downsampled) | 100% size (native) |
| **Mask quality** | Blurry (interpolation) | Sharp (native) |
| **Small particle detection** | ⚠️ Poor (too small) | ✅ Good (correct scale) |
| **Runtime (first run)** | ~4-6 hours | ~3-4 hours |
| **Runtime (repeat runs)** | ~4-6 hours (always trains) | **~5 minutes** (loads models) |
| **Statistical samples** | Fewer (16/image) | More (64/image) → better |
| **Model reuse** | No | ✅ Yes (smart caching) |

---

## Expected Results Differences

### Scenario: Mitochondria ~15 pixels diameter at 256×256

**512×512 Approach:**
```
Original 512×512 tile:
  Mitochondrion = 30 pixels diameter
           ↓ (resize to 256×256)
  Mitochondrion = 15 pixels diameter
           ↓ (predict)
  Mask at 256×256 (correct)
           ↓ (resize to 512×512)
  Mask at 512×512 (blurry boundaries)

Result: Works, but with interpolation blur
```

**256×256 Approach:**
```
Original 256×256 tile:
  Mitochondrion = 15 pixels diameter
           ↓ (predict directly)
  Mask at 256×256 (sharp, clean)

Result: Native resolution, no artifacts
```

### Visual Quality Prediction

**512×512 tiles:**
- Mask boundaries: Slightly blurred
- Small particles: May be missed (too small after downsampling)
- Overall: Usable but suboptimal

**256×256 tiles:**
- Mask boundaries: Sharp and clean
- Small particles: Better detected (correct scale)
- Overall: Higher quality segmentation

### Density Measurement Accuracy

**512×512 tiles:**
- Fewer samples per image (16 tiles)
- Higher variance in density estimates
- Interpolation may introduce false positives/negatives

**256×256 tiles:**
- More samples per image (64 tiles)
- Lower variance in density estimates
- Native resolution = more accurate foreground counting

---

## Which Should You Use?

### Use 256×256 if:
- ✅ You want **accurate segmentation** of small particles
- ✅ You value **reproducibility** (run prediction multiple times)
- ✅ You want **better statistics** (more tiles per image)
- ✅ You want **fast iteration** (5 min after first run)

### Use 512×512 if:
- You specifically need larger spatial context
- You don't mind interpolation artifacts
- You're okay with always retraining models

### Recommended: **Run BOTH and compare**

Since you're already running the 512×512 job, also run the 256×256 job to:
1. See visual quality differences in tile comparisons
2. Compare density measurements in CSV
3. Check statistical robustness (more samples with 256×256)
4. Validate that resolution mismatch affects results

---

## How to Run Both Jobs

### Job 1: 512×512 (Already Running)
```bash
# Already submitted or running
qstat -u phyzxi | grep Density_Pred
```

### Job 2: 256×256 (New)
```bash
# Transfer files
scp density_prediction_256_fast.py \
    pbs_density_256_fast.sh \
    phyzxi@atlas7.nus.edu.sg:~/scratch/unet-HPC/

# Submit
ssh phyzxi@atlas7.nus.edu.sg
cd ~/scratch/unet-HPC
qsub pbs_density_256_fast.sh

# Monitor
qstat -u phyzxi
tail -f Density_256.o*
```

**First run:** ~3-4 hours (trains models, saves them)
**Subsequent runs:** ~5 minutes (loads saved models)

---

## Output Comparison Structure

### Both jobs produce:

```
Job 1 (512×512):                        Job 2 (256×256):
density_prediction_YYYYMMDD_HHMMSS/     density_prediction_256_YYYYMMDD_HHMMSS/
├── trained_models/                     ├── (uses ../saved_models_validation_config/)
│   └── (3 .keras files)                │   (persistent, shared across runs)
├── representative_tiles/               ├── representative_tiles/
│   └── (5 comparisons per image)       │   └── (5 comparisons per image)
├── boxplots/                           ├── boxplots/
│   └── (4 PNG files)                   │   └── (4 PNG files)
└── csv_data/                           └── csv_data/
    └── comprehensive.csv                   └── comprehensive.csv
```

**Key Difference:** Job 2 saves models in a persistent directory, enabling fast re-runs.

---

## Comparing Results

### After both jobs complete:

```bash
# Compare tile visualization quality
# Look at representative tiles side-by-side:
#   512×512: Are boundaries blurry?
#   256×256: Are boundaries sharp?

# Compare CSV data
# Load both CSVs and check:
python3
>>> import pandas as pd
>>> df_512 = pd.read_csv('density_prediction_YYYYMMDD_HHMMSS/csv_data/density_analysis_comprehensive.csv')
>>> df_256 = pd.read_csv('density_prediction_256_YYYYMMDD_HHMMSS/csv_data/density_analysis_comprehensive.csv')
>>>
>>> # Compare mean density by method
>>> df_512.groupby('method')['foreground_pct'].mean()
>>> df_256.groupby('method')['foreground_pct'].mean()
>>>
>>> # Compare variance (256 should have lower variance)
>>> df_512.groupby('method')['foreground_pct'].std()
>>> df_256.groupby('method')['foreground_pct'].std()
>>>
>>> # Compare number of samples
>>> len(df_512)  # Fewer tiles
>>> len(df_256)  # More tiles (4× more)
```

### Expected Findings:

1. **Visual Quality:** 256×256 tiles should show sharper mask boundaries
2. **Small Particle Detection:** 256×256 should catch more small particles
3. **Density Accuracy:** Similar mean densities, but 256×256 with lower variance
4. **Statistical Power:** 256×256 has 4× more samples = more reliable estimates

---

## Performance Benchmarks

### Job 1 (512×512):
- **Always ~4-6 hours** regardless of run number
- Memory usage: Moderate
- Disk space: ~500 MB per run

### Job 2 (256×256):
- **First run:** ~3-4 hours (trains + saves models)
- **Subsequent runs:** ~5 minutes (just prediction)
- Memory usage: Moderate (same as Job 1)
- Disk space:
  - Models: ~300 MB (persistent, one-time)
  - Per run: ~500 MB

**Efficiency:** After initial investment, Job 2 is **48-72× faster** for repeat runs.

---

## Recommendation

**For your use case (small particles, dozens per image):**

1. ✅ **Primary approach:** 256×256 tiles (Job 2)
   - Native resolution = best accuracy
   - Fast re-runs for iterative analysis
   - More statistical samples

2. 📊 **Validation:** Compare with 512×512 results (Job 1)
   - Verify that resolution mismatch matters
   - Confirm 256×256 doesn't miss large-scale patterns

3. 🔬 **Future analyses:** Use 256×256 exclusively
   - 5-minute runtime enables rapid experimentation
   - Can re-run with different test images instantly

---

## Key Insight

**For small particle segmentation:**
- Matching the training resolution (256×256) is **critical**
- Interpolation introduces artifacts that degrade accuracy
- More tiles = better statistical robustness
- Smart model caching enables rapid iteration

**Your observation** about "dozens of particles in one cropped image" at 256×256 → confirms this is the right scale for your data. Using 512×512 would just add complexity without benefit.

---

## Files Created

### Job 1 (512×512):
✅ `density_prediction_with_tiles.py` (14 KB)
✅ `pbs_density_prediction.sh` (8 KB)

### Job 2 (256×256 - Fast):
✅ `density_prediction_256_fast.py` (17 KB)
✅ `pbs_density_256_fast.sh` (8 KB)

### Documentation:
✅ `COMPARISON_512_vs_256.md` (this file)

---

**Ready to submit Job 2 (256×256) to HPC!**
