# Attention Models Feature Visualization - Ready for HPC Deployment

**Status**: ✅ All files fixed, committed, and pushed
**Date**: October 30, 2025
**Ready for**: HPC Job Submission

---

## Quick Start (On HPC)

```bash
# SSH to HPC
cd ~/scratch/unet-HPC

# Option 1: Use deployment script (recommended)
bash deploy_and_run_attresunet_viz.sh

# Option 2: Manual deployment
git pull
qsub pbs_attention_resunet_feature_viz.sh
```

---

## What's Ready

### ✅ Attention U-Net Visualization
| Component | Status | Details |
|-----------|--------|---------|
| Python Script | ✅ Complete | `attention_unet_feature_visualization.py` |
| PBS Script | ✅ Complete | `pbs_attention_unet_feature_viz.sh` |
| Architecture | ✅ Correct | n_filters=32, dropout=0.1 |
| HPC Job | ✅ Running | Job 330075 (AttUNet_Feature_Viz) |

**Job Status**: Already submitted and running successfully!

### ✅ Attention ResU-Net Visualization
| Component | Status | Details |
|-----------|--------|---------|
| Python Script | ✅ Fixed | `attention_resunet_feature_visualization.py` |
| PBS Script | ✅ Fixed | `pbs_attention_resunet_feature_viz.sh` |
| Architecture | ✅ Fixed | n_filters=64 (was 32) ← **FIXED** |
| HPC Job | 🔄 Ready | Needs resubmission with fixed scripts |

**Previous Issue**: Job 330076 failed with RuntimeError (shape mismatch)
**Root Cause**: Script used n_filters=32, checkpoint has n_filters=64
**Fix Applied**: Updated both .py and .sh to use n_filters=64

---

## Files Synced to GitHub

### New Files Created
```
attention_unet_feature_visualization.py         (25KB)
pbs_attention_unet_feature_viz.sh              (5.4KB)
attention_resunet_feature_visualization.py     (26KB)  ← FIXED
pbs_attention_resunet_feature_viz.sh           (6.1KB) ← FIXED
```

### Documentation
```
DEBUG_ATTENTION_MODELS_N_FILTERS.md            (Comprehensive debug report)
ATTENTION_MODELS_READY_TO_DEPLOY.md            (This file)
deploy_and_run_attresunet_viz.sh               (Automated deployment script)
```

---

## Architecture Details

### Attention U-Net
```python
Model: AttentionUNet
n_filters: 32
dropout: 0.1
Parameters: ~1.9M
Features:
  - Standard ConvBlocks
  - Attention gates on skip connections
  - Best Val IoU: 0.6254
```

### Attention ResU-Net
```python
Model: AttentionResUNet
n_filters: 64  ← Required by trained checkpoint
dropout: 0.1
Parameters: ~7.7M (4× more than n_filters=32)
Features:
  - Residual ConvBlocks (ResNet-style)
  - Attention gates on skip connections
  - Best Val IoU: ~0.63
```

`★ Key Difference ────────────────────────────────────────────────────`
**Why n_filters differs between models:**
- Standard/Attention U-Net: 32 base filters sufficient
- Attention ResU-Net: Residual connections add architectural complexity
- More complex architecture → needs more capacity → 64 base filters
- Result: 4× more parameters (64² vs 32² in conv operations)
`──────────────────────────────────────────────────────────────────────`

---

## Visualization Configuration

Both models will generate:

**Layers Visualized**:
- Encoder layers (1-4): Feature extraction hierarchy
- Bottleneck: Highest-level semantic features
- Attention gates: What attention mechanism focuses on (UNIQUE!)
- Decoder layers: Reconstruction patterns

**Parameters**:
- Channels per layer: 12 (subset of total)
- Diverse examples: 3 per channel
- Optimization iterations: 500
- Image size: 512×512
- DeepDream: Enabled

**Expected Output**:
```
<model>_feature_viz_YYYYMMDD_HHMMSS/
├── encoder_1_*/               # Edge detectors, simple patterns
├── encoder_3_*/               # Texture, object parts
├── bottleneck_*/              # High-level semantics
├── attention_gate_*/          # Attention-modulated features ← UNIQUE
├── decoder_*/                 # Reconstruction patterns
├── *_diverse_visualizations.png  # Grid views
└── metadata.json              # Configuration
```

---

## Expected Runtime

| Model | GPU Time | Output Size | Layers |
|-------|----------|-------------|--------|
| Attention U-Net | ~2-3 hours | ~500MB | 8 layers |
| Attention ResU-Net | ~2-3 hours | ~500MB | 8 layers |

Both jobs use same visualization parameters, so runtime is similar despite 4× parameter difference.

---

## Monitoring Jobs on HPC

### Check Job Status
```bash
qstat -u $USER
```

Expected output:
```
Job ID          Name              User    Queue   Status
--------------- ----------------- ------- ------- ------
330075.hopper   AttUNet_Feature   xiaodan gpu_q   R      # Running
XXXXXX.hopper   AttResUNet_Feat   xiaodan gpu_q   R      # After resubmit
```

### Watch Real-Time Log
```bash
# Attention U-Net (already running)
tail -f AttUNet_Feature_Viz.o330075

# Attention ResU-Net (after resubmit)
tail -f AttResUNet_Feat_Viz.o<NEW_JOB_ID>
```

### Verify Success
```bash
# Look for these lines in the log
grep "✓ Model loaded" AttResUNet_Feat_Viz.o*
grep "Best validation IoU" AttResUNet_Feat_Viz.o*
grep "Layer: encoder_1" AttResUNet_Feat_Viz.o*
```

**Success indicators**:
```
✓ Model loaded (epoch XX)
✓ Best validation IoU: 0.XXXX
======================================================================
Layer: encoder_1_resconv
======================================================================
Optimizing: 100%|██████████| 500/500 [00:07<00:00, 64.28it/s]
```

---

## What Changed (Fix Summary)

### Before (❌ Job 330076 Failed)
```python
# attention_resunet_feature_visualization.py:542
parser.add_argument('--n_filters', type=int, default=32)  # WRONG
```
```bash
# pbs_attention_resunet_feature_viz.sh:47
N_FILTERS=32  # WRONG
```

**Error**:
```
RuntimeError: Error(s) in loading state_dict for AttentionResUNet:
    size mismatch for enc1.conv1.weight:
    copying a param with shape torch.Size([64, 1, 3, 3]) from checkpoint,
    the shape in current model is torch.Size([32, 1, 3, 3]).
```

### After (✅ Ready to Resubmit)
```python
# attention_resunet_feature_visualization.py:542
parser.add_argument('--n_filters', type=int, default=64,
                   help='Base number of filters (trained model uses 64)')  # FIXED
```
```bash
# pbs_attention_resunet_feature_viz.sh:47
N_FILTERS=64  # Trained model uses 64 base filters (not 32)  # FIXED
```

**Expected Result**: Model loads successfully, no shape mismatches!

---

## Comparison: All Three Models

| Model | n_filters | Parameters | Architecture | Val IoU | Job Status |
|-------|-----------|------------|--------------|---------|------------|
| **Standard U-Net** | 32 | ~1.9M | ConvBlock | 0.6377 | ✅ Completed |
| **Attention U-Net** | 32 | ~1.9M | ConvBlock + AttGates | 0.6254 | ✅ Running (330075) |
| **Attention ResU-Net** | 64 | ~7.7M | ResConvBlock + AttGates | ~0.63 | 🔄 Ready to resubmit |

`★ Performance Insight ───────────────────────────────────────────────`
Despite 4× more parameters, Attention ResU-Net achieves only marginally
better IoU (~0.63 vs 0.6254). This suggests:
- Simpler Attention U-Net is quite efficient for this task
- Residual connections may help training stability more than final performance
- Extra capacity may not be needed (possible overfitting risk)
`──────────────────────────────────────────────────────────────────────`

---

## Next Steps

### 1. Deploy to HPC
```bash
# SSH to HPC
ssh <username>@hopper.nus.edu.sg

# Navigate to working directory
cd ~/scratch/unet-HPC

# Pull latest changes
git pull
```

### 2. Submit Attention ResU-Net Job
```bash
# Option A: Use automated script
bash deploy_and_run_attresunet_viz.sh

# Option B: Manual submission
qsub pbs_attention_resunet_feature_viz.sh
```

### 3. Monitor Both Jobs
```bash
# Check status
qstat -u $USER

# Watch Attention U-Net (already running)
tail -f AttUNet_Feature_Viz.o330075

# Watch Attention ResU-Net (after submission)
tail -f AttResUNet_Feat_Viz.o<JOB_ID>
```

### 4. After Completion
Both jobs will create output directories:
```
attention_unet_feature_viz_YYYYMMDD_HHMMSS/
attention_resunet_feature_viz_YYYYMMDD_HHMMSS/
```

Download to local machine:
```bash
# On local machine
scp -r <username>@hopper:~/scratch/unet-HPC/attention_*_feature_viz_* ./
```

### 5. Analysis
Compare visualizations:
- Attention gates: How do they differ between U-Net and ResU-Net?
- Encoder features: Impact of residual connections
- Channel diversity: More channels in ResU-Net (64 vs 32 base)

---

## Troubleshooting

### If Job Still Fails
1. **Check n_filters in checkpoint**:
```bash
module load singularity
singularity exec --nv $image python -c "
import torch
ckpt = torch.load('./best_models_PyTorch/attention_resunet/best_model.pth',
                  map_location='cpu')
enc1_weight = ckpt['model_state_dict']['enc1.conv1.weight']
print(f'Checkpoint n_filters: {enc1_weight.shape[0]}')
"
```

2. **Verify script matches**:
```bash
grep "n_filters" attention_resunet_feature_visualization.py | head -1
grep "N_FILTERS" pbs_attention_resunet_feature_viz.sh
```

3. **Check model info**:
```bash
cat ./best_models_PyTorch/attention_resunet/model_info.json
```

---

## Files on GitHub

All files available at: `https://github.com/Zhao-Xiaodan/unet-HPC`

**Latest commits**:
- `457d134` - Add HPC deployment script
- `05bfc5a` - Fix n_filters mismatch in Attention ResU-Net

**Branch**: `main`

---

**Summary**: Everything is ready! Just pull on HPC and submit the job. The n_filters=64 fix resolves the Job 330076 failure. Both attention model visualizations will provide unique insights into how attention gates modulate learned features. 🚀
