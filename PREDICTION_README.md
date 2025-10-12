# Prediction and Density Analysis

## Overview

This workflow performs inference on test images using trained U-Net, ResU-Net, and Attention ResU-Net models, and calculates particle density using the CLAHE+OTSU reference method.

## Files

### 1. `predict_with_density_analysis.py`

Main Python script for prediction and analysis.

**Features:**
- Loads best trained models for each architecture (BS=8, combined loss functions)
- Extracts 512×512 tiles from test images
- Performs prediction on each tile
- Calculates particle density using two methods:
  - **CLAHE+OTSU**: Reference method (from `Particle-density-calculation.py`)
  - **Predicted masks**: Direct calculation from model outputs
- Generates boxplot comparisons for each test image
- Exports comprehensive summary statistics

**Input:**
- `./test_images/`: Directory containing test images (.tif, .tiff, .png)
- `./hyperparam_comprehensive_20251012_005054/`: Directory with trained models (.hdf5)

**Output:**
- `prediction_analysis_YYYYMMDD_HHMMSS/`
  - `predicted_masks/unet/`: U-Net predictions
  - `predicted_masks/resunet/`: ResU-Net predictions
  - `predicted_masks/attention_resunet/`: Attention ResU-Net predictions
  - `boxplots/`: Density comparison plots for each test image
  - `summary/`: Summary statistics and overall comparison plot

### 2. `pbs_predict_density.sh`

PBS job submission script for HPC execution.

**Configuration:**
- Walltime: 4 hours
- Resources: 1 GPU, 36 CPUs, 240GB memory
- TensorFlow container: `tensorflow_2.16.1-cuda_12.5.0_24.06.sif`

## Usage

### On HPC Cluster

1. **Ensure test images are in place:**
   ```bash
   ls test_images/
   # Should show .tif or .png files
   ```

2. **Verify trained models exist:**
   ```bash
   ls hyperparam_comprehensive_20251012_005054/*.hdf5
   # Should show model files for unet, resunet, attention_resunet
   ```

3. **Submit job:**
   ```bash
   qsub pbs_predict_density.sh
   ```

4. **Monitor job:**
   ```bash
   qstat -u $USER
   tail -f Predict_Density_Analysis.o*
   ```

5. **Check results:**
   ```bash
   ls -R prediction_analysis_*/
   ```

### Local Testing (if models available)

```bash
# Activate environment
conda activate unetCNN

# Run prediction
python predict_with_density_analysis.py
```

## Output Structure

```
prediction_analysis_20251012_150000/
├── predicted_masks/
│   ├── unet/
│   │   ├── testimage1_tile000_y0_x0.png
│   │   ├── testimage1_tile001_y0_x512.png
│   │   └── ...
│   ├── resunet/
│   │   └── ...
│   └── attention_resunet/
│       └── ...
├── boxplots/
│   ├── testimage1_density_comparison.png
│   ├── testimage2_density_comparison.png
│   └── ...
└── summary/
    ├── density_analysis_summary.csv
    └── overall_density_comparison.png
```

## Density Calculation Methods

### Method 1: CLAHE+OTSU (Reference)

Following `Particle-density-calculation.py` methodology:

1. **Rescale image to full 0-255 range:**
   ```
   I_rescaled = 255 × (I - I_min) / (I_max - I_min)
   ```

2. **Apply CLAHE (Contrast Limited Adaptive Histogram Equalization):**
   ```
   clipLimit = 2.0
   tileGridSize = (8, 8)
   I_clahe = CLAHE(I_rescaled)
   ```

3. **Apply OTSU thresholding (inverse):**
   ```
   threshold = OTSU(I_clahe)
   binary_mask = (I_clahe < threshold) × 255
   # White = particles (255), Black = background (0)
   ```

4. **Calculate density:**
   ```
   density = (binary_mask > 0).sum() / total_pixels
   ```

### Method 2: Predicted Masks (Model-based)

1. **Model prediction:**
   ```
   prediction = model.predict(normalized_tile)
   binary_mask = (prediction > 0.5) × 255
   ```

2. **Calculate density:**
   ```
   density = (binary_mask > 0).sum() / total_pixels
   ```

## Model Selection

The script automatically searches for the best available model for each architecture with the following priority:

1. `model_{arch}_bs8_dr0.3_combined_tversky.hdf5` (Best from search)
2. `model_{arch}_bs8_dr0.3_combined.hdf5` (Alternative)
3. `model_{arch}_bs8_dr0.3_*.hdf5` (Any BS=8 model)
4. `model_{arch}_bs*_dr*_*.hdf5` (Any model for this architecture)

Where `{arch}` is one of: `unet`, `resunet`, `attention_resunet`

## Best Models (from Hyperparameter Search)

Based on `hyperparam_comprehensive_20251012_005054` results:

| Architecture | Model File | Peak Jaccard | Loss Function |
|--------------|------------|--------------|---------------|
| **ResU-Net** | `model_resunet_bs8_dr0.3_combined_tversky.hdf5` | **0.307** | Combined Tversky |
| Attention ResU-Net | `model_attention_resunet_bs8_dr0.3_focal_tversky.hdf5` | 0.264 | Focal Tversky |
| U-Net | `model_unet_bs8_dr0.3_focal_tversky.hdf5` | 0.245 | Focal Tversky |

Alternative with "combined" loss (if combined_tversky not available):

| Architecture | Model File | Peak Jaccard | Loss Function |
|--------------|------------|--------------|---------------|
| Attention ResU-Net | `model_attention_resunet_bs8_dr0.3_combined.hdf5` | 0.249 | Combined (Dice+Focal) |
| U-Net | `model_unet_bs8_dr0.3_combined.hdf5` | 0.231 | Combined (Dice+Focal) |

## Expected Output

### Summary CSV Format

```csv
image,architecture,mean_density,std_density,median_density,min_density,max_density,num_tiles
testimage1,clahe_otsu,0.1234,0.0456,0.1200,0.0500,0.2100,12
testimage1,unet,0.1150,0.0420,0.1100,0.0450,0.1950,12
testimage1,resunet,0.1180,0.0435,0.1120,0.0480,0.2000,12
testimage1,attention_resunet,0.1160,0.0425,0.1110,0.0460,0.1980,12
...
```

### Boxplot Features

Each boxplot (`testimage_density_comparison.png`) shows:
- Box: 25th-75th percentile (IQR)
- Whiskers: 1.5×IQR range
- Individual points: Each tile's density
- Statistics overlay: Mean (μ) and standard deviation (σ) for each method

### Overall Comparison Plot

4-panel figure showing:
1. Mean density comparison across architectures
2. Standard deviation (variability) comparison
3. Median density comparison
4. Maximum density comparison

All metrics aggregated across all test images.

## Troubleshooting

### Issue: Models not found

**Solution:**
- Verify models were saved during training (check directory size)
- If models missing, re-run hyperparameter search with model saving enabled
- Alternatively, train specific models:
  ```bash
  # Re-train best configuration
  python train_best_model.py --architecture resunet --batch-size 8 \
         --loss combined_tversky --epochs 100
  ```

### Issue: Out of memory during prediction

**Solution:**
- Reduce batch size for tile prediction in the script
- Modify `predict_on_tiles()` to process tiles in smaller batches
- Increase GPU memory limits in PBS script

### Issue: Test images not loading

**Solution:**
- Check image format (should be .tif, .tiff, or .png)
- Verify images are grayscale or will be converted correctly
- Check file permissions

## Performance Expectations

Based on hardware and typical test image sizes:

- **Small images** (2048×2048): ~4 tiles, 1-2 minutes per architecture
- **Medium images** (4096×4096): ~16 tiles, 3-5 minutes per architecture
- **Large images** (8192×8192): ~64 tiles, 10-15 minutes per architecture

For typical test set (10-20 images, mixed sizes) with 3 architectures:
- **Total runtime:** 30-90 minutes
- **Output size:** 50-200 MB (depending on number of tiles)

## Analysis Interpretation

### Density Values

Typical density ranges for microbeads:
- **Low density:** 0.05-0.10 (sparse)
- **Medium density:** 0.10-0.20 (moderate)
- **High density:** 0.20-0.35 (dense)

### Comparing Methods

**CLAHE+OTSU vs Predicted:**
- CLAHE+OTSU is the reference method (ground truth)
- Good model predictions should correlate strongly with CLAHE+OTSU
- Systematic differences indicate model bias:
  - **Predicted < CLAHE+OTSU**: Model under-segments (misses particles)
  - **Predicted > CLAHE+OTSU**: Model over-segments (false detections)

### Architecture Comparison

Based on hyperparameter search results:
- **ResU-Net**: Expected to show highest density correlation with reference
- **Attention ResU-Net**: May show better performance on challenging regions
- **U-Net**: Baseline performance, may under-perform on small particles

## References

1. Hyperparameter search results: `hyperparam_comprehensive_20251012_005054/COMPREHENSIVE_SEARCH_REPORT.md`
2. Loss functions: `loss_functions.py`
3. Model architectures: `model_architectures.py`
4. Density calculation reference: `Particle-density-calculation.py`

## Contact

For issues or questions about this workflow, refer to the comprehensive search report or the main project documentation.

---

**Last Updated:** 2025-10-12
**Version:** 1.0
**Compatible with:** TensorFlow 2.16.1, Python 3.8+
