# Individual Model Training Scripts

## Overview

This document describes the **3 pairs of training scripts** for individual model architectures with hyperparameter search. Each model can be trained independently in parallel on the HPC cluster.

## Files Created

### Pair 1: Standard UNet
- **Python script:** `train_unet_hyperparam.py`
- **PBS script:** `pbs_train_unet.sh`
- **Job name:** `UNet_Hyperparam`
- **Walltime:** 24 hours
- **Output:** `unet_hyperparam_YYYYMMDD_HHMMSS/`

### Pair 2: Attention UNet
- **Python script:** `train_attention_unet_hyperparam.py`
- **PBS script:** `pbs_train_attention_unet.sh`
- **Job name:** `AttentionUNet_Hyperparam`
- **Walltime:** 36 hours
- **Output:** `attention_unet_hyperparam_YYYYMMDD_HHMMSS/`

### Pair 3: Attention ResUNet
- **Python script:** `train_attention_resunet_hyperparam.py`
- **PBS script:** `pbs_train_attention_resunet.sh`
- **Job name:** `AttentionResUNet_Hyperparam`
- **Walltime:** 48 hours
- **Output:** `attention_resunet_hyperparam_YYYYMMDD_HHMMSS/`

## Key Features

All three scripts share the same architecture:

1. **✅ NO Lambda layers** - Use `RepeatElements` custom layer for proper serialization
2. **✅ Saves BOTH best and final models**
   - Best model: `checkpoints/{experiment_name}/best_model.keras`
   - Final model: `models/{experiment_name}_final.keras`
3. **✅ Hyperparameter grid search**
4. **✅ Proper BinaryFocalLoss serialization**
5. **✅ Early stopping** (patience=20)
6. **✅ Learning rate reduction** (patience=10)
7. **✅ CSV logging** of training history

## Hyperparameter Grid

All three models use the same hyperparameter search space:

```python
'hyperparam_grid': {
    'n_filters': [16, 32, 64],        # Base number of filters
    'dropout': [0.1, 0.2, 0.3],       # Dropout rate
    'batch_norm': [True],              # Batch normalization (always on)
    'learning_rate': [0.001, 0.003, 0.005],  # Learning rate
}
```

**Total combinations per model:** 3 × 3 × 1 × 3 = **27 experiments**

## Usage

### Option 1: Train All Models in Parallel

Submit all three jobs simultaneously to train models in parallel:

```bash
# Submit all three jobs
qsub pbs_train_unet.sh
qsub pbs_train_attention_unet.sh
qsub pbs_train_attention_resunet.sh

# Monitor all jobs
qstat -u $USER

# Check logs
tail -f UNet_Hyperparam.o*
tail -f AttentionUNet_Hyperparam.o*
tail -f AttentionResUNet_Hyperparam.o*
```

**Advantage:** All models train simultaneously, total time = longest job (~48 hours)

### Option 2: Train Models Sequentially

Submit jobs one at a time:

```bash
# Train standard UNet first
qsub pbs_train_unet.sh

# After UNet completes, train Attention UNet
qsub pbs_train_attention_unet.sh

# After Attention UNet completes, train Attention ResUNet
qsub pbs_train_attention_resunet.sh
```

**Advantage:** Uses fewer cluster resources at once

### Option 3: Train Only Specific Model

Train just one model:

```bash
# Train only Attention UNet
qsub pbs_train_attention_unet.sh
```

## Dataset

All scripts use the same dataset:
- **Images:** `./dataset_shrunk_masks/images/` (98 images)
- **Masks:** `./dataset_shrunk_masks/masks/` (98 masks)
- **Train/Val split:** 80/20 (78 train, 20 validation)
- **Random seed:** 42 (for reproducibility)

## Expected Runtime

### Per Model

| Model | Experiments | Avg Time per Exp | Total Time | Walltime |
|-------|-------------|------------------|------------|----------|
| **UNet** | 27 | ~30-45 min | ~12-20 hours | 24 hours |
| **Attention UNet** | 27 | ~40-60 min | ~18-27 hours | 36 hours |
| **Attention ResUNet** | 27 | ~50-80 min | ~22-36 hours | 48 hours |

### All Models in Parallel

**Total runtime:** ~48 hours (longest job)

**Cluster resources:** 3 × (1 GPU + 36 CPUs + 240GB RAM) = 3 GPUs simultaneously

### All Models Sequentially

**Total runtime:** ~60-90 hours

**Cluster resources:** 1 GPU + 36 CPUs + 240GB RAM

## Output Structure

Each model produces its own output directory:

```
{model_name}_hyperparam_YYYYMMDD_HHMMSS/
├── CONFIG.json                                # Configuration used
├── {model_name}_results.csv                   # All experiment results
│
├── models/                                    # Final models
│   ├── {model}_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001_final.keras
│   ├── {model}_n_filters16_dropout0p1_batch_normTrue_learning_rate0p003_final.keras
│   └── ... (27 final models)
│
├── checkpoints/                               # Best models
│   ├── {model}_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001/
│   │   └── best_model.keras
│   ├── {model}_n_filters16_dropout0p1_batch_normTrue_learning_rate0p003/
│   │   └── best_model.keras
│   └── ... (27 checkpoint directories)
│
└── logs/                                      # Training histories
    ├── {model}_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001_history.csv
    ├── {model}_n_filters16_dropout0p1_batch_normTrue_learning_rate0p003_history.csv
    └── ... (27 CSV logs)
```

## Results Analysis

### During Training

Monitor progress:

```bash
# Check latest results for UNet
tail -20 unet_hyperparam_*/unet_results.csv

# Check latest results for Attention UNet
tail -20 attention_unet_hyperparam_*/attention_unet_results.csv

# Check latest results for Attention ResUNet
tail -20 attention_resunet_hyperparam_*/attention_resunet_results.csv
```

### After Training

Compare best models across architectures:

```python
import pandas as pd

# Load results
unet_results = pd.read_csv('unet_hyperparam_*/unet_results.csv')
attn_unet_results = pd.read_csv('attention_unet_hyperparam_*/attention_unet_results.csv')
attn_resunet_results = pd.read_csv('attention_resunet_hyperparam_*/attention_resunet_results.csv')

# Find best for each
best_unet = unet_results.nlargest(1, 'best_val_iou')
best_attn_unet = attn_unet_results.nlargest(1, 'best_val_iou')
best_attn_resunet = attn_resunet_results.nlargest(1, 'best_val_iou')

# Compare
print("Best UNet:", best_unet['best_val_iou'].values[0])
print("Best Attention UNet:", best_attn_unet['best_val_iou'].values[0])
print("Best Attention ResUNet:", best_attn_resunet['best_val_iou'].values[0])
```

## Model Loading

After training, load best models:

```python
from tensorflow import keras
from models_fixed import RepeatElements
from loss_functions_fixed import BinaryFocalLoss, jacard_coef, dice_coef

custom_objects = {
    'RepeatElements': RepeatElements,
    'BinaryFocalLoss': BinaryFocalLoss,
    'jacard_coef': jacard_coef,
    'dice_coef': dice_coef,
}

# Load best UNet
unet_model = keras.models.load_model(
    'unet_hyperparam_*/checkpoints/{best_experiment}/best_model.keras',
    custom_objects=custom_objects
)

# Load best Attention UNet
attn_unet_model = keras.models.load_model(
    'attention_unet_hyperparam_*/checkpoints/{best_experiment}/best_model.keras',
    custom_objects=custom_objects
)

# Load best Attention ResUNet
attn_resunet_model = keras.models.load_model(
    'attention_resunet_hyperparam_*/checkpoints/{best_experiment}/best_model.keras',
    custom_objects=custom_objects
)
```

## Comparison with Combined Script

### Combined Script (`train_attention_models_hyperparam.py`)
- Trains Attention UNet + Attention ResUNet sequentially
- 2 architectures × 18 combinations = 36 experiments
- Total runtime: ~48-60 hours
- 1 job, 1 output directory

### Individual Scripts (This Set)
- Train each model independently
- 3 architectures × 27 combinations = 81 experiments total
- Parallel runtime: ~48 hours (if run simultaneously)
- 3 jobs, 3 output directories

### Key Differences

| Aspect | Combined Script | Individual Scripts |
|--------|----------------|-------------------|
| **Architectures** | 2 (no standard UNet) | 3 (includes standard UNet) |
| **Combinations per model** | 18 (2×3×1×3) | 27 (3×3×1×3) |
| **n_filters grid** | [16, 32] | [16, 32, 64] |
| **Parallelization** | Sequential only | Can run in parallel |
| **Flexibility** | All or nothing | Train any subset |
| **Output organization** | 1 directory | 3 separate directories |

## Advantages of Individual Scripts

1. **Parallelization:** Run all 3 models simultaneously on different GPUs
2. **Flexibility:** Train only specific model(s) of interest
3. **Fault tolerance:** If one job fails, others continue
4. **Better organization:** Separate results for each architecture
5. **Includes standard UNet:** Baseline comparison model
6. **Larger hyperparameter search:** 27 vs 18 combinations per model

## Submission Strategy Recommendations

### For Maximum Speed (Parallel)
```bash
# Submit all at once
qsub pbs_train_unet.sh && \
qsub pbs_train_attention_unet.sh && \
qsub pbs_train_attention_resunet.sh
```
**Time:** ~48 hours
**GPUs needed:** 3 simultaneously

### For Resource Conservation (Sequential)
```bash
# Submit with dependencies
JOB1=$(qsub pbs_train_unet.sh)
JOB2=$(qsub -W depend=afterok:$JOB1 pbs_train_attention_unet.sh)
JOB3=$(qsub -W depend=afterok:$JOB2 pbs_train_attention_resunet.sh)
```
**Time:** ~60-90 hours
**GPUs needed:** 1 at a time

### For Baseline Comparison
```bash
# Train only standard UNet first
qsub pbs_train_unet.sh

# After reviewing results, decide on attention models
```

## Troubleshooting

### Job Fails Immediately

Check PBS output file:
```bash
cat UNet_Hyperparam.o*
```

Common issues:
- Dataset path wrong (should be `./dataset_shrunk_masks/`)
- Missing `models_fixed.py` or `loss_functions_fixed.py`
- TensorFlow container not found

### Out of Memory

If GPU OOM occurs, reduce batch size:

Edit Python script:
```python
CONFIG = {
    # ...
    'batch_size': 2,  # Reduce from 4 to 2
    # ...
}
```

Or reduce filter grid:
```python
'hyperparam_grid': {
    'n_filters': [16, 32],  # Remove 64
    # ...
}
```

### Early Stopping Too Soon

Increase patience:
```python
CONFIG = {
    # ...
    'early_stopping_patience': 30,  # Increase from 20
    # ...
}
```

## Summary

**Created 3 pairs of scripts for independent model training:**

1. **Standard UNet** - Baseline model
2. **Attention UNet** - Attention mechanisms
3. **Attention ResUNet** - Residual + attention

**Each script:**
- ✅ NO Lambda layer issues
- ✅ Saves best AND final models
- ✅ 27 hyperparameter combinations
- ✅ Can run in parallel or sequentially

**Next steps:**
1. Submit jobs: `qsub pbs_train_{model}.sh`
2. Monitor: `qstat -u $USER`
3. Analyze results: Compare `{model}_results.csv`
4. Use best models for density analysis

---

**Created:** October 16, 2025
**Author:** Claude Code
**Status:** ✅ Ready for HPC submission
