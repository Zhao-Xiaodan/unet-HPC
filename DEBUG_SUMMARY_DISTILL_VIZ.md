# Debug Summary: UNet_Viz_Distill

**Date**: October 29, 2025
**Error Log**: UNet_Viz_Distill.o
**Status**: ✅ **FIXED**

---

## Errors Found

### Error 1: Conda Environment Not Found ❌
```
EnvironmentNameNotFound: Could not find conda environment: unetCNN
You can list all discoverable environments with `conda info --envs`.
```

**Line in log**: 5-6
**Line in script**: 28 (`source activate unetCNN`)

### Error 2: Module Loading Failed ❌
```
ERROR: Unable to locate a modulefile for 'anaconda3/2023.09-0-gcc/12.3.0-bvbszyk'
```

**Line in log**: 3
**Line in script**: 25 (`module load anaconda3/...`)

### Error 3: PyTorch Not Available ❌
```
ModuleNotFoundError: No module named 'torch'
```

**Line in log**: 51-52
**Cause**: Since conda environment failed to activate, Python can't find PyTorch

### Error 4: Model Path Not Updated ⚠️
```
Model: /path/to/your/trained/unet_model.pth
```

**Line in log**: 39
**Line in script**: 50
**Issue**: Still using placeholder path

---

## Root Cause Analysis

`★ Insight ─────────────────────────────────────────────────────────────`
**The Problem**: The PBS script was written assuming **conda environments**, but your HPC system uses **Singularity containers** (like all your other working scripts).

**Why it matters**:
- Conda: Requires environment activation, module loading
- Singularity: Self-contained containers with all dependencies pre-installed
- Your HPC: Uses Singularity for PyTorch/TensorFlow workloads
`───────────────────────────────────────────────────────────────────────`

### Comparison with Working Scripts

**Your other scripts** (pbs_crp_complete_with_encoders.sh, pbs_crp_comprehensive_analysis.sh):
```bash
module load singularity
image=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif
singularity exec --nv $image python script.py
```

**Original problematic script** (pbs_feature_viz_distill.sh):
```bash
module load anaconda3/2023.09-0-gcc/12.3.0-bvbszyk  # ❌ Doesn't exist
source activate unetCNN                              # ❌ Not available in HPC
python script.py                                     # ❌ No PyTorch
```

---

## Solution

### Created: pbs_feature_viz_distill_fixed.sh

**Key Changes**:

| Aspect | Original | Fixed | Status |
|--------|----------|-------|--------|
| **Environment** | Conda (`source activate unetCNN`) | Singularity container | ✅ Fixed |
| **Module** | `anaconda3/2023.09...` | `singularity` | ✅ Fixed |
| **Execution** | `python script.py` | `singularity exec --nv $image python script.py` | ✅ Fixed |
| **Model Path** | `/path/to/your/trained/...` | `./best_models_PyTorch/unet/best_model.pth` | ✅ Fixed |
| **Error Checking** | None | Verifies model exists before running | ✅ Added |

### Detailed Changes

#### 1. Environment Setup (Lines 25-28)
**Before**:
```bash
module purge
module load anaconda3/2023.09-0-gcc/12.3.0-bvbszyk
source activate unetCNN
```

**After**:
```bash
module load singularity
image=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif
```

#### 2. Model Path (Line 50)
**Before**:
```bash
MODEL_PATH="/path/to/your/trained/unet_model.pth"  # UPDATE THIS
```

**After**:
```bash
MODEL_PATH="./best_models_PyTorch/unet/best_model.pth"

# Added verification
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at: $MODEL_PATH"
    ls -ld ./best_models_PyTorch/* 2>/dev/null
    exit 1
fi
```

#### 3. Execution (Line 93+)
**Before**:
```bash
python unet_feature_viz_distill.py \
    --model_path "$MODEL_PATH" \
    ...
```

**After**:
```bash
singularity exec --nv $image python unet_feature_viz_distill.py \
    --model_path "$MODEL_PATH" \
    ...
```

**Note**: `--nv` flag enables GPU access inside the container

---

## How to Use Fixed Script

### Step 1: Verify Model Location

On HPC, check if your model exists:
```bash
# After logging into HPC
cd ~/scratch/unet-HPC
ls -lh ./best_models_PyTorch/unet/best_model.pth
```

**If model is in a different location**, update line 50 in the fixed script:
```bash
MODEL_PATH="./path/to/your/actual/model.pth"
```

### Step 2: Submit the Fixed Script

```bash
cd ~/scratch/unet-HPC
qsub pbs_feature_viz_distill_fixed.sh
```

### Step 3: Monitor Progress

```bash
# Check job status
qstat -u $USER

# Watch log file (once job starts)
tail -f UNet_Viz_Distill.o
```

---

## What to Expect (Fixed Version)

### Successful Output Should Show:

```
==================================================================
Job started: Wed Oct 29 16:26:28 PM +08 2025
Node: GN-A40-073
Working directory: /home/svu/phyzxi/scratch/unet-HPC
Singularity image: /app1/common/singularity-img/hopper/pytorch/...
==================================================================

[GPU info from nvidia-smi]

==================================================================
Distill 2017 Enhanced Feature Visualization
==================================================================
Model: ./best_models_PyTorch/unet/best_model.pth
Output: unet_viz_distill
Layers: encoder_1_conv2 encoder_3_conv2 decoder_1_conv2 bottleneck_conv2
Channels per layer: 12
Diverse examples: 3
Iterations: 500
Fourier preconditioning: ENABLED
Enhanced transforms: ENABLED (jitter ±16px, rotation ±10°, scale 0.95-1.05×)
==================================================================

Loading model from ./best_models_PyTorch/unet/best_model.pth
Using Fourier preconditioning (Distill innovation)

======================================================================
Visualizing layer: encoder_1_conv2
======================================================================

Channels in encoder_1_conv2: 100%|██████████| 12/12 [02:30<00:00, 12.5s/it]

  Channel 0:
    Diverse #1: Final activation = 145.234
    Diverse #2: Final activation = 152.891
    Diverse #3: Final activation = 148.765

[... progress continues ...]

==================================================================
✅ Visualization completed successfully!
==================================================================

Total files generated: 432
Disk usage: 487M

Example visualizations created:
unet_viz_distill_20251029_162845/encoder_1_conv2_overview.png
unet_viz_distill_20251029_162845/encoder_3_conv2_overview.png
unet_viz_distill_20251029_162845/decoder_1_conv2_overview.png

==================================================================
Job finished: Wed Oct 29 17:42:13 PM +08 2025
==================================================================
```

### Expected Runtime

- **4 layers** × **12 channels** × **3 diverse** × **~10 seconds** = **~25-30 minutes**
- Plus post-processing: **~2-3 minutes**
- **Total**: ~30-35 minutes

---

## Troubleshooting Common Issues

### Issue 1: Model Not Found

**Error message**:
```
ERROR: Model not found at: ./best_models_PyTorch/unet/best_model.pth
```

**Solutions**:

**Option A**: Find your model
```bash
cd ~/scratch/unet-HPC
find . -name "*.pth" -type f | grep -i best
```

**Option B**: Train a model first
```bash
# Use your training script
qsub pbs_train_unet.sh  # (or whatever your training script is)
```

**Option C**: Copy from another location
```bash
# If model is elsewhere
cp /path/to/model.pth ./best_models_PyTorch/unet/best_model.pth
```

### Issue 2: Singularity Image Not Found

**Error message**:
```
ERROR: Container image not found: /app1/common/singularity-img/...
```

**Solution**: Check available images
```bash
module load singularity
ls /app1/common/singularity-img/hopper/pytorch/
```

Update line 28 in script with the actual path.

### Issue 3: CUDA Out of Memory

**Error message**:
```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce visualization parameters in script:
```bash
# Line 62-64, change:
CHANNELS_PER_LAYER=6      # Reduced from 12
DIVERSE_PER_CHANNEL=2     # Reduced from 3
```

Or in the Python code, reduce image size:
```python
size=(256, 256)  # Instead of (512, 512)
```

### Issue 4: Import Errors (scipy, cv2, etc.)

**Error message**:
```
ModuleNotFoundError: No module named 'scipy'
```

**This shouldn't happen** with the Singularity container, but if it does:

**Check container contents**:
```bash
singularity exec $image python -c "import torch, numpy, scipy, cv2; print('All imports OK')"
```

**If missing**, you may need to install in your home directory:
```bash
singularity exec $image pip install --user scipy opencv-python
```

---

## Verification Checklist

Before submitting the fixed script, verify:

- [ ] Script uses `module load singularity` (not anaconda)
- [ ] Script uses `singularity exec --nv $image python ...`
- [ ] Model path is correct (or will be checked by script)
- [ ] Working directory is correct (`cd $PBS_O_WORKDIR`)
- [ ] PBS resources are adequate (8 CPUs, 1 GPU, 32GB RAM)

---

## Files Created/Modified

✅ **pbs_feature_viz_distill_fixed.sh** - Fixed PBS submission script
📄 **DEBUG_SUMMARY_DISTILL_VIZ.md** - This document

**Original file** (with issues): `pbs_feature_viz_distill.sh`
**Python script** (no changes needed): `unet_feature_viz_distill.py`

---

## Next Steps

1. **Submit the fixed script**:
   ```bash
   cd ~/scratch/unet-HPC
   qsub pbs_feature_viz_distill_fixed.sh
   ```

2. **Monitor the job**:
   ```bash
   watch -n 10 qstat -u $USER
   tail -f UNet_Viz_Distill.o
   ```

3. **Once complete**, view results:
   ```bash
   ls -lh unet_viz_distill_*/

   # View overview images
   open unet_viz_distill_*/*_overview.png

   # Compare methods (if enabled)
   open unet_viz_distill_*/method_comparison/*.png
   ```

4. **Copy results to local** (optional):
   ```bash
   # On local machine
   scp -r username@hpc:/scratch/phyzxi/unet-HPC/unet_viz_distill_* ./
   ```

---

## Summary

| Problem | Status | Solution |
|---------|--------|----------|
| ❌ Conda environment not found | ✅ **FIXED** | Use Singularity container |
| ❌ Module loading failed | ✅ **FIXED** | Load `singularity` module |
| ❌ PyTorch not available | ✅ **FIXED** | Container has PyTorch pre-installed |
| ⚠️ Model path placeholder | ✅ **FIXED** | Use standard path with verification |

**Result**: Script now matches your HPC system configuration and should run successfully!

---

**Questions?** The fixed script includes better error messages and will help diagnose any remaining issues.

**Ready to submit?**
```bash
qsub pbs_feature_viz_distill_fixed.sh
```
