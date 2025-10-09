# HPC Prediction Setup Guide

## Quick Start - 3 Steps

### Step 1: Prepare Test Images on HPC

```bash
# SSH to HPC
ssh phyzxi@hpc

# Navigate to project directory
cd ~/scratch/unet-HPC

# Create test image directory
mkdir -p test_image

# Option A: Copy images from your dataset
cp dataset_microscope/images/*.tif test_image/

# Option B: Upload from local machine (from your local terminal)
scp -r /path/to/local/images/* phyzxi@hpc:~/scratch/unet-HPC/test_image/

# Verify images are there
ls test_image/
# Should show your .tif, .png, or .jpg files
```

### Step 2: Submit Prediction Job

```bash
# Make sure you're in the project directory
cd ~/scratch/unet-HPC

# Submit the job
qsub pbs_predict_microscope.sh

# You should see: 278XXX.stdct-mgmt-02
```

### Step 3: Monitor and Retrieve Results

```bash
# Check job status
qstat -u phyzxi

# Watch job output (while running)
tail -f Microscope_Prediction.o<JOBID>

# After completion, download results to your local machine
# (Run this from your LOCAL terminal, not HPC)
scp -r phyzxi@hpc:~/scratch/unet-HPC/predictions_* ./local_predictions/
```

---

## Detailed Configuration

### Customize Prediction Parameters

Edit `pbs_predict_microscope.sh` (lines 16-23):

```bash
# INPUT/OUTPUT DIRECTORIES
INPUT_DIR="./test_image"                           # Your images here
OUTPUT_DIR="./predictions_$(date +%Y%m%d_%H%M%S)" # Auto-timestamped
MODEL_DIR="./microscope_training_20251008_074915" # Your training results

# PREDICTION PARAMETERS
TILE_SIZE=256        # Must match training (256)
OVERLAP=32           # 32=fast, 64=high quality
THRESHOLD=0.5        # 0.3-0.7 range
```

### Parameter Recommendations

| Image Size | Tile Size | Overlap | Expected Time (GPU) |
|------------|-----------|---------|---------------------|
| 256×256 | 256 | 16 | ~1 second |
| 512×512 | 256 | 32 | ~2 seconds |
| 1920×1080 | 256 | 32 | ~15 seconds |
| 3840×2160 | 256 | 32 | ~45 seconds |
| 3840×2160 | 256 | 64 | ~60 seconds (better quality) |

**For best quality on large images:** Use `OVERLAP=64`

**For faster processing:** Use `OVERLAP=16`

---

## Resource Requirements

### Current PBS Settings

```bash
#PBS -l walltime=02:00:00              # 2 hours (adjust if needed)
#PBS -l select=1:ncpus=12:ngpus=1:mem=64gb
```

### Adjust Based on Your Needs

**For small batches (<20 images, <1920×1080):**
```bash
#PBS -l walltime=00:30:00              # 30 minutes
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
```

**For large batches (>100 images or 4K resolution):**
```bash
#PBS -l walltime=04:00:00              # 4 hours
#PBS -l select=1:ncpus=12:ngpus=1:mem=96gb
```

**CPU-only mode (no GPU available):**
```bash
#PBS -l select=1:ncpus=24:mem=64gb    # Remove ngpus
# Expect 5-10× slower processing
```

---

## Output Structure on HPC

After job completes:

```
~/scratch/unet-HPC/
├── predictions_20251009_143022/
│   ├── masks/
│   │   ├── image001_mask.png
│   │   └── ...
│   ├── overlays/
│   │   ├── image001_overlay.png
│   │   └── ...
│   ├── comparisons/
│   │   ├── image001_comparison.png
│   │   └── ...
│   ├── prediction_summary.txt        # Statistics
│   └── prediction_log.txt             # Full console output
└── Microscope_Prediction.o278XXX      # PBS job log
```

---

## Troubleshooting on HPC

### Issue 1: No images found

```bash
ERROR: No image files found in ./test_image
```

**Solution:**
```bash
# Check directory contents
ls -la test_image/

# Verify file extensions are supported
# Supported: .tif, .tiff, .png, .jpg, .jpeg, .bmp

# Check file permissions
chmod 644 test_image/*
```

### Issue 2: Model not found

```bash
ERROR: Model directory not found
```

**Solution:**
```bash
# List available training directories
ls -d microscope_training_*

# Update MODEL_DIR in pbs_predict_microscope.sh to match
# Edit line 18:
MODEL_DIR="./microscope_training_YYYYMMDD_HHMMSS"
```

### Issue 3: Job runs out of time

```bash
# In PBS output: killed due to walltime exceeded
```

**Solution:**
```bash
# Edit walltime in pbs_predict_microscope.sh line 2:
#PBS -l walltime=04:00:00    # Increase to 4 hours

# Or process images in smaller batches
```

### Issue 4: Out of memory

```bash
ResourceExhaustedError: OOM when allocating tensor
```

**Solution:**
```bash
# Option 1: Reduce overlap (in pbs_predict_microscope.sh)
OVERLAP=16    # Instead of 32

# Option 2: Request more memory (in PBS header)
#PBS -l select=1:ncpus=12:ngpus=1:mem=128gb
```

### Issue 5: predictions are all black

```bash
Predicted mitochondria coverage: 0.00%
```

**Possible causes:**
1. Model predicting all background (validation collapse issue)
2. Threshold too high

**Solutions:**
```bash
# Try lower threshold (edit pbs_predict_microscope.sh)
THRESHOLD=0.3    # Instead of 0.5

# Or try THRESHOLD=0.2 for very faint predictions
```

---

## Advanced Usage Examples

### Example 1: Process Multiple Directories

Create a script `batch_predict.sh`:

```bash
#!/bin/bash

# Directories to process
DIRS=("experiment_1/images" "experiment_2/images" "experiment_3/images")

for dir in "${DIRS[@]}"; do
    echo "Processing $dir..."

    # Modify INPUT_DIR in PBS script
    sed -i "s|INPUT_DIR=.*|INPUT_DIR=\"$dir\"|" pbs_predict_microscope.sh

    # Submit job
    qsub pbs_predict_microscope.sh

    # Wait 10 seconds between submissions
    sleep 10
done
```

### Example 2: Process with Different Thresholds

```bash
# Test multiple thresholds to find optimal value
for threshold in 0.3 0.4 0.5 0.6; do
    sed -i "s|THRESHOLD=.*|THRESHOLD=$threshold|" pbs_predict_microscope.sh
    sed -i "s|OUTPUT_DIR=.*|OUTPUT_DIR=\"./predictions_thresh_${threshold}\"|" pbs_predict_microscope.sh
    qsub pbs_predict_microscope.sh
    sleep 5
done
```

### Example 3: High-Quality Mode for Publication

Edit `pbs_predict_microscope.sh`:

```bash
# Maximum quality settings
TILE_SIZE=256
OVERLAP=96          # Very high overlap
THRESHOLD=0.5

# Request more resources
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=12:ngpus=1:mem=96gb
```

---

## Downloading Results to Local Machine

### Option 1: Download Specific Output

```bash
# From your LOCAL terminal (not HPC):

# Download just the masks
scp -r phyzxi@hpc:~/scratch/unet-HPC/predictions_20251009_143022/masks ./

# Download comparisons for presentation
scp -r phyzxi@hpc:~/scratch/unet-HPC/predictions_20251009_143022/comparisons ./

# Download summary file
scp phyzxi@hpc:~/scratch/unet-HPC/predictions_20251009_143022/prediction_summary.txt ./
```

### Option 2: Download Everything

```bash
# Download entire prediction directory
scp -r phyzxi@hpc:~/scratch/unet-HPC/predictions_20251009_143022 ./
```

### Option 3: Compress First (for large datasets)

```bash
# On HPC:
cd ~/scratch/unet-HPC
tar -czf predictions_20251009_143022.tar.gz predictions_20251009_143022/

# On local machine:
scp phyzxi@hpc:~/scratch/unet-HPC/predictions_20251009_143022.tar.gz ./
tar -xzf predictions_20251009_143022.tar.gz
```

---

## Monitoring Job Progress

### Real-time Monitoring

```bash
# Method 1: Watch PBS output file
tail -f Microscope_Prediction.o<JOBID>

# Method 2: Check prediction log (after job starts)
tail -f predictions_*/prediction_log.txt

# Method 3: Count processed files
watch -n 5 'ls predictions_*/masks/ | wc -l'
```

### Check Job Status

```bash
# Basic status
qstat -u phyzxi

# Detailed status
qstat -f <JOBID>

# Check which node job is running on
qstat -n <JOBID>

# View job history
qstat -x <JOBID>
```

---

## Integration with Analysis Workflow

After predictions complete, analyze on HPC:

```bash
# Create analysis script: analyze_predictions.py
cat > analyze_predictions.py << 'EOF'
import cv2
import numpy as np
import os
from pathlib import Path

# Process all masks
mask_dir = "predictions_20251009_143022/masks"
results = []

for mask_file in Path(mask_dir).glob("*.png"):
    mask = cv2.imread(str(mask_file), 0)

    # Count mitochondria
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # Statistics
    total_area = np.sum(mask > 0)
    coverage = (total_area / mask.size) * 100
    num_mito = num_labels - 1

    results.append({
        'file': mask_file.name,
        'coverage_%': coverage,
        'num_mitochondria': num_mito,
        'total_area_px': total_area
    })

    print(f"{mask_file.name}: {num_mito} mitochondria, {coverage:.2f}% coverage")

# Save summary
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('analysis_summary.csv', index=False)
print(f"\nSaved analysis to analysis_summary.csv")
print(f"Mean coverage: {df['coverage_%'].mean():.2f}%")
print(f"Mean count: {df['num_mitochondria'].mean():.1f}")
EOF

# Run analysis
module load singularity
singularity exec --nv \
    /app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif \
    python3 analyze_predictions.py
```

---

## File Checklist Before Submission

Make sure these files exist on HPC:

- ✅ `pbs_predict_microscope.sh` (PBS job script)
- ✅ `predict_microscope.py` (Python prediction script)
- ✅ `models.py` or `224_225_226_models.py` (model definitions)
- ✅ `focal_loss.py` (or will be auto-created)
- ✅ `microscope_training_20251008_074915/` (directory with trained models)
- ✅ `test_image/` (directory with images to process)

Verify with:
```bash
ls -l pbs_predict_microscope.sh predict_microscope.py models.py
ls -d microscope_training_* test_image/
```

---

## Summary

**To run predictions on HPC:**

1. **Setup** (one-time):
   ```bash
   mkdir -p test_image
   # Copy or upload images to test_image/
   ```

2. **Submit**:
   ```bash
   qsub pbs_predict_microscope.sh
   ```

3. **Monitor**:
   ```bash
   tail -f Microscope_Prediction.o<JOBID>
   ```

4. **Download**:
   ```bash
   # From local machine:
   scp -r phyzxi@hpc:~/scratch/unet-HPC/predictions_* ./
   ```

**Processing time estimates:**
- Small images (256×256): ~1s per image
- Medium images (1920×1080): ~15s per image
- Large images (3840×2160): ~45s per image
- Batch of 100 large images: ~1.5 hours

All done with automated quality checks, progress logging, and comprehensive outputs! 🚀
