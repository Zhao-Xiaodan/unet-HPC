# Best Models Cache - PyTorch

**Directory:** `./best_models_PyTorch/`
**Purpose:** Cache the best performing models for fast prediction without re-searching experiment directories

---

## Overview

This directory serves as a **persistent cache** for the best-performing PyTorch models from your comparison experiments. Once populated, predictions can run immediately without scanning through all experiment checkpoints.

## Structure

```
best_models_PyTorch/
├── unet/
│   ├── best_model.pth           # Model checkpoint
│   └── model_info.json          # Hyperparameters and metadata
├── attention_unet/
│   ├── best_model.pth
│   └── model_info.json
└── attention_resunet/
    ├── best_model.pth
    └── model_info.json
```

## How It Works

### First Run

When you run `predict_pytorch_comparison.py` for the first time:

1. **Check cache:** Looks for `./best_models_PyTorch/`
2. **Not found:** Searches experiment directory `all_results.csv`
3. **Finds best models:** Based on highest validation IoU per architecture
4. **Copies to cache:**
   - Copies `best_model.pth` from experiment checkpoints
   - Saves `model_info.json` with hyperparameters
5. **Future runs:** Load directly from cache ✅

### Subsequent Runs

When cache exists with all 3 architectures:

1. **Check cache:** `./best_models_PyTorch/` exists
2. **Verify completeness:** All 3 architectures have `best_model.pth` + `model_info.json`
3. **Load from cache:** Instant loading without experiment directory search
4. **Time saved:** ~5-10 seconds per run

## Example Cache Files

### `unet/model_info.json`

```json
{
  "n_filters": 32,
  "dropout": 0.2,
  "learning_rate": 0.001,
  "best_val_iou": 0.6377,
  "model_name": "unet_n_filters32_dropout0.2_learning_rate0.001",
  "source_experiment": "pytorch_comparison_no_aug_20251021_121918",
  "cached_date": "2025-10-22 14:30:15"
}
```

### `unet/best_model.pth`

Binary PyTorch checkpoint containing:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `best_val_iou`: Best validation IoU achieved
- `history`: Training history
- `epoch`: Epoch number when best IoU achieved

**Size:** ~50-200 MB per architecture (depends on n_filters)

## When to Rebuild Cache

Rebuild the cache when:

1. **New experiments complete** with better performance
2. **Different experiment source** desired
3. **Cache corruption** suspected

### How to Rebuild

**Option 1: Delete cache directory**

```bash
rm -rf ./best_models_PyTorch/
# Next run will rebuild from scratch
```

**Option 2: Delete specific architecture**

```bash
rm -rf ./best_models_PyTorch/unet/
# Next run will rebuild only UNet
```

**Option 3: Update cache programmatically**

```bash
# Run prediction with different experiment directory
python predict_pytorch_comparison.py \
    --experiment pytorch_comparison_with_aug_20251021_122018 \
    --output predictions_new \
    --test_images ./test_images

# This will update cache with models from the new experiment
```

## Verifying Cache

Check cache status:

```bash
# Check if cache exists
ls -lh best_models_PyTorch/

# Should show:
# drwxr-xr-x  4 user  staff  128B Oct 22 14:30 unet
# drwxr-xr-x  4 user  staff  128B Oct 22 14:30 attention_unet
# drwxr-xr-x  4 user  staff  128B Oct 22 14:30 attention_resunet

# Check model sizes
du -sh best_models_PyTorch/*/best_model.pth

# Should show ~50-200MB per file
```

Verify metadata:

```bash
# View UNet metadata
cat best_models_PyTorch/unet/model_info.json | python -m json.tool

# Check all validation IoUs
for arch in unet attention_unet attention_resunet; do
    echo -n "$arch: "
    python -c "import json; print(json.load(open('best_models_PyTorch/$arch/model_info.json'))['best_val_iou'])"
done
```

## Benefits

### Performance

| Operation | Without Cache | With Cache |
|-----------|--------------|------------|
| Load models | ~10-15 sec | ~2-3 sec |
| Disk seeks | 100+ files | 6 files |
| CSV parsing | Required | Not needed |

### Reliability

- **No search failures:** Models always in known location
- **No experiment dependency:** Can delete old experiment directories
- **Version tracking:** `model_info.json` records source experiment

### Reproducibility

- **Fixed model set:** Same models used across all predictions
- **Metadata preserved:** Hyperparameters documented
- **Source tracked:** Know which experiment produced best models

## Cache vs Experiment Directory

### Experiment Directory

```
pytorch_comparison_no_aug_20251021_121918/
├── all_results.csv                    # 81 rows (all models)
├── unet/checkpoints/
│   ├── unet_n_filters16_dropout0.1_lr0.001/best_model.pth
│   ├── unet_n_filters16_dropout0.1_lr0.003/best_model.pth
│   ├── ... (27 total UNet models)
├── attention_unet/checkpoints/
│   └── ... (27 total Attention UNet models)
└── attention_resunet/checkpoints/
    └── ... (27 total Attention ResUNet models)
```

**Total:** 81 models × ~100MB = ~8GB

### Cache Directory

```
best_models_PyTorch/
├── unet/best_model.pth                # 1 best model
├── attention_unet/best_model.pth      # 1 best model
└── attention_resunet/best_model.pth   # 1 best model
```

**Total:** 3 models × ~100MB = ~300MB

**Space saved:** ~96% reduction

## Multiple Experiment Comparison

To compare predictions from different experiments:

```bash
# Option 1: Use different output directories
python predict_pytorch_comparison.py \
    --experiment pytorch_comparison_no_aug_20251021_121918 \
    --output predictions_no_aug

python predict_pytorch_comparison.py \
    --experiment pytorch_comparison_with_aug_20251021_122018 \
    --output predictions_with_aug

# Cache will update to the latest experiment
# To preserve: Rename cache between runs

# Option 2: Rename cache between runs
mv best_models_PyTorch best_models_PyTorch_no_aug
python predict_pytorch_comparison.py \
    --experiment pytorch_comparison_with_aug_20251021_122018 \
    --output predictions_with_aug
mv best_models_PyTorch best_models_PyTorch_with_aug
```

## Maintenance

### Regular Checks

```bash
# Monthly: Verify cache integrity
for arch in unet attention_unet attention_resunet; do
    if [ ! -f "best_models_PyTorch/$arch/best_model.pth" ]; then
        echo "ERROR: Missing checkpoint for $arch"
    fi
    if [ ! -f "best_models_PyTorch/$arch/model_info.json" ]; then
        echo "ERROR: Missing metadata for $arch"
    fi
done
```

### Backup

```bash
# Backup cache (recommended before major changes)
tar -czf best_models_PyTorch_backup_$(date +%Y%m%d).tar.gz best_models_PyTorch/

# Restore from backup
tar -xzf best_models_PyTorch_backup_YYYYMMDD.tar.gz
```

## Troubleshooting

### Cache incomplete message

**Symptom:** "⚠ Cache incomplete, searching experiment directory..."

**Cause:** One or more architectures missing from cache

**Solution:**
```bash
# Check what's missing
ls best_models_PyTorch/

# Rebuild cache
rm -rf best_models_PyTorch/
# Re-run prediction script
```

### Wrong models loaded

**Symptom:** Predictions don't match expected experiment

**Solution:**
```bash
# Check cache source
cat best_models_PyTorch/*/model_info.json | grep source_experiment

# If wrong, delete and rebuild
rm -rf best_models_PyTorch/
```

### Corrupted checkpoint

**Symptom:** Model loading fails with PyTorch error

**Solution:**
```bash
# Delete corrupted architecture
rm -rf best_models_PyTorch/<architecture>/

# Re-run will rebuild that architecture only
```

---

**Note:** This cache directory is automatically managed by `predict_pytorch_comparison.py`. Manual editing is not required but is safe if needed.

**Last Updated:** October 22, 2025
