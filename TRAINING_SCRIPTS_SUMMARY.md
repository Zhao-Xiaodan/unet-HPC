# Training Scripts Summary

## Quick Reference Table

| Model | Python Script | PBS Script | Job Name | Walltime | Experiments | Submit Command |
|-------|---------------|------------|----------|----------|-------------|----------------|
| **Standard UNet** | `train_unet_hyperparam.py` | `pbs_train_unet.sh` | `UNet_Hyperparam` | 24h | 27 | `qsub pbs_train_unet.sh` |
| **Attention UNet** | `train_attention_unet_hyperparam.py` | `pbs_train_attention_unet.sh` | `AttentionUNet_Hyperparam` | 36h | 27 | `qsub pbs_train_attention_unet.sh` |
| **Attention ResUNet** | `train_attention_resunet_hyperparam.py` | `pbs_train_attention_resunet.sh` | `AttentionResUNet_Hyperparam` | 48h | 27 | `qsub pbs_train_attention_resunet.sh` |
| **Combined (Old)** | `train_attention_models_hyperparam.py` | `pbs_train_attention_hyperparam.sh` | `Attention_Hyperparam` | 48h | 36 | `qsub pbs_train_attention_hyperparam.sh` |

## Hyperparameter Grid Comparison

| Parameter | Combined Script | Individual Scripts | Notes |
|-----------|----------------|-------------------|-------|
| `n_filters` | [16, 32] | [16, 32, 64] | Individual has more options |
| `dropout` | [0.1, 0.2, 0.3] | [0.1, 0.2, 0.3] | Same |
| `batch_norm` | [True] | [True] | Same (always on) |
| `learning_rate` | [0.001, 0.003, 0.005] | [0.001, 0.003, 0.005] | Same |
| **Total combos** | 18 per model | 27 per model | 50% more exploration |

## Quick Start Commands

### Train All Models in Parallel (Recommended)
```bash
# Submit all three jobs at once
qsub pbs_train_unet.sh
qsub pbs_train_attention_unet.sh
qsub pbs_train_attention_resunet.sh

# Monitor all jobs
watch -n 10 'qstat -u $USER'
```

### Train All Models Sequentially (Resource-Conservative)
```bash
# Submit with dependencies
JOB1=$(qsub pbs_train_unet.sh | cut -d'.' -f1)
JOB2=$(qsub -W depend=afterok:$JOB1 pbs_train_attention_unet.sh | cut -d'.' -f1)
qsub -W depend=afterok:$JOB2 pbs_train_attention_resunet.sh
```

### Train Only One Model
```bash
# Just Attention UNet (for example)
qsub pbs_train_attention_unet.sh
```

## Output Directories

```
unet_hyperparam_20251016_HHMMSS/
├── CONFIG.json
├── unet_results.csv
├── models/ (27 final models)
├── checkpoints/ (27 best models)
└── logs/ (27 CSV histories)

attention_unet_hyperparam_20251016_HHMMSS/
├── CONFIG.json
├── attention_unet_results.csv
├── models/ (27 final models)
├── checkpoints/ (27 best models)
└── logs/ (27 CSV histories)

attention_resunet_hyperparam_20251016_HHMMSS/
├── CONFIG.json
├── attention_resunet_results.csv
├── models/ (27 final models)
├── checkpoints/ (27 best models)
└── logs/ (27 CSV histories)
```

## Log Files

```bash
# PBS output logs
UNet_Hyperparam.o{JOBID}
AttentionUNet_Hyperparam.o{JOBID}
AttentionResUNet_Hyperparam.o{JOBID}

# Console logs (with full training output)
train_unet_hyperparam_console_20251016_HHMMSS.log
train_attention_unet_hyperparam_console_20251016_HHMMSS.log
train_attention_resunet_hyperparam_console_20251016_HHMMSS.log
```

## Monitoring Commands

```bash
# Check job status
qstat -u $USER

# Watch logs live
tail -f UNet_Hyperparam.o*
tail -f AttentionUNet_Hyperparam.o*
tail -f AttentionResUNet_Hyperparam.o*

# Check results so far
tail -20 unet_hyperparam_*/unet_results.csv
tail -20 attention_unet_hyperparam_*/attention_unet_results.csv
tail -20 attention_resunet_hyperparam_*/attention_resunet_results.csv

# Count completed models
ls -1 unet_hyperparam_*/models/*.keras | wc -l
ls -1 attention_unet_hyperparam_*/models/*.keras | wc -l
ls -1 attention_resunet_hyperparam_*/models/*.keras | wc -l
```

## Resource Usage

### Per Job
- **CPUs:** 36
- **GPUs:** 1 × A40
- **Memory:** 240 GB
- **Storage:** ~5-10 GB per model (depends on filters)

### All Jobs in Parallel
- **CPUs:** 108 total
- **GPUs:** 3 × A40
- **Memory:** 720 GB total
- **Storage:** ~15-30 GB total

## Expected Timeline

### Parallel Execution (3 jobs simultaneously)
```
Hour 0:   All jobs start
Hour 24:  UNet completes (27 models)
Hour 36:  Attention UNet completes (27 models)
Hour 48:  Attention ResUNet completes (27 models)
Total:    ~48 hours, 81 models total
```

### Sequential Execution (1 job at a time)
```
Hour 0:   UNet starts
Hour 24:  UNet completes, Attention UNet starts
Hour 60:  Attention UNet completes, Attention ResUNet starts
Hour 108: Attention ResUNet completes
Total:    ~108 hours, 81 models total
```

## Syntax Verification

All scripts have been syntax-checked:
```bash
✓ train_unet_hyperparam.py
✓ train_attention_unet_hyperparam.py
✓ train_attention_resunet_hyperparam.py
```

## Files Checklist

Before submitting, verify these files exist:

```bash
# Training scripts (Python)
✓ train_unet_hyperparam.py
✓ train_attention_unet_hyperparam.py
✓ train_attention_resunet_hyperparam.py

# PBS scripts
✓ pbs_train_unet.sh
✓ pbs_train_attention_unet.sh
✓ pbs_train_attention_resunet.sh

# Required modules
✓ models_fixed.py
✓ loss_functions_fixed.py

# Dataset
✓ dataset_shrunk_masks/images/ (98 images)
✓ dataset_shrunk_masks/masks/ (98 masks)
```

## Next Steps After Training

1. **Identify best models:**
   ```bash
   # Sort by best_val_iou
   sort -t',' -k4 -nr unet_hyperparam_*/unet_results.csv | head -5
   sort -t',' -k4 -nr attention_unet_hyperparam_*/attention_unet_results.csv | head -5
   sort -t',' -k4 -nr attention_resunet_hyperparam_*/attention_resunet_results.csv | head -5
   ```

2. **Compare across architectures:**
   ```python
   import pandas as pd

   unet = pd.read_csv('unet_hyperparam_*/unet_results.csv')
   attn_unet = pd.read_csv('attention_unet_hyperparam_*/attention_unet_results.csv')
   attn_resunet = pd.read_csv('attention_resunet_hyperparam_*/attention_resunet_results.csv')

   all_results = pd.concat([unet, attn_unet, attn_resunet])
   best_overall = all_results.nlargest(10, 'best_val_iou')
   print(best_overall[['model', 'best_val_iou', 'n_filters', 'dropout', 'learning_rate']])
   ```

3. **Use best models for density analysis**

4. **Generate training visualizations**

---

**Created:** October 16, 2025
**Status:** ✅ Ready for submission
