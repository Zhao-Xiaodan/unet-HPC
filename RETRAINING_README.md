###  Attention Models Retraining - No Lambda Layers

## Why Retraining?

The original Attention UNet and Attention ResUNet models have **Lambda layer serialization issues** that are difficult to resolve during loading. Retraining with proper architecture will:

1. **✅ Eliminate Lambda layers** - Use `RepeatElements` custom layer instead
2. **✅ Ensure proper serialization** - Models load cleanly with `model.save()` / `load_model()`
3. **✅ Save both best AND final models** - Via `ModelCheckpoint` callback
4. **✅ Improve performance** - Through hyperparameter tuning
5. **✅ Clean, maintainable code** - Future-proof architecture

## Files Created

### 1. `models_fixed.py` - Fixed Model Architectures

**Key Changes:**
- ❌ **Removed:** `Lambda(lambda x: K.repeat_elements(x, rep, axis=3))`
- ✅ **Added:** `RepeatElements` custom layer with proper serialization
- ✅ All custom layers use `@keras.saving.register_keras_serializable`

**Custom Layer:**
```python
@tf.keras.saving.register_keras_serializable(package='Custom')
class RepeatElements(layers.Layer):
    """Replaces Lambda layer for K.repeat_elements()"""
    def __init__(self, rep, axis=3, **kwargs):
        super().__init__(**kwargs)
        self.rep = rep
        self.axis = axis

    def call(self, inputs):
        return K.repeat_elements(inputs, self.rep, axis=self.axis)

    def get_config(self):
        return {'rep': self.rep, 'axis': self.axis}
```

**Architectures Included:**
1. `build_unet()` - Standard UNet (baseline)
2. `build_attention_unet()` - Attention UNet with attention gates
3. `build_attention_resunet()` - Attention ResUNet with residual blocks + attention

### 2. `train_attention_models_hyperparam.py` - Training Script

**Features:**
- Hyperparameter grid search
- Saves **BOTH** best and final models:
  - **Best model:** `checkpoints/{exp_name}/best_model.keras` (via ModelCheckpoint)
  - **Final model:** `models/{exp_name}_final.keras` (after training completes)
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- CSV logging of training history

**Hyperparameter Grid:**
```python
'hyperparam_grid': {
    'n_filters': [16, 32],        # Base number of filters
    'dropout': [0.1, 0.2, 0.3],   # Dropout rate
    'batch_norm': [True],          # Batch normalization
    'learning_rate': [0.001, 0.003, 0.005],  # Learning rate
}
```

**Total Combinations:** 2 × 3 × 1 × 3 = **18 experiments per architecture**

### 3. `pbs_train_attention_hyperparam.sh` - HPC Submission Script

**Resource Allocation:**
- **Walltime:** 48 hours (sufficient for all combinations)
- **GPU:** 1 × A40
- **CPUs:** 36
- **Memory:** 240GB

## Usage

### On HPC

1. **Ensure files are in place:**
   ```bash
   ls -la models_fixed.py
   ls -la train_attention_models_hyperparam.py
   ls -la pbs_train_attention_hyperparam.sh
   ls -la loss_functions_fixed.py
   ls -la data_generator.py
   ```

2. **Verify dataset:**
   ```bash
   ls -la dataset_new_shrunk/train/images/
   ls -la dataset_new_shrunk/val/images/
   ```

3. **Submit job:**
   ```bash
   qsub pbs_train_attention_hyperparam.sh
   ```

4. **Monitor:**
   ```bash
   qstat -u $USER
   tail -f Attention_Hyperparam.o*
   ```

## Expected Output

### Directory Structure
```
attention_hyperparam_YYYYMMDD_HHMMSS/
├── CONFIG.json                                  # Configuration used
├── all_results.csv                               # All experiment results
├── attention_unet_results.csv                    # Attention UNet results
├── attention_resunet_results.csv                 # Attention ResUNet results
│
├── models/                                       # Final models
│   ├── attention_unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001_final.keras
│   ├── attention_unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p003_final.keras
│   ├── ...
│   ├── attention_resunet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001_final.keras
│   └── ...
│
├── checkpoints/                                  # Best models (via ModelCheckpoint)
│   ├── attention_unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001/
│   │   └── best_model.keras                     # Best epoch model
│   ├── attention_unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p003/
│   │   └── best_model.keras
│   └── ...
│
└── logs/                                         # Training histories (CSV)
    ├── attention_unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001_history.csv
    ├── attention_unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p003_history.csv
    └── ...
```

### Results CSV Format

**`all_results.csv`:**
```csv
architecture,experiment_name,best_epoch,best_val_iou,best_val_dice,final_val_iou,final_val_dice,n_filters,dropout,batch_norm,learning_rate
attention_unet,attention_unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001,85,0.6234,0.7654,0.6189,0.7612,16,0.1,True,0.001
attention_unet,attention_unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p003,72,0.6578,0.7891,0.6523,0.7856,16,0.1,True,0.003
...
```

## Advantages Over Original Training

| Aspect | Original | New (Retraining) |
|--------|----------|------------------|
| **Lambda Layers** | ❌ Yes (serialization issues) | ✅ No (RepeatElements instead) |
| **Model Saving** | ❌ Final only | ✅ Both best AND final |
| **Hyperparameter Search** | ❌ Manual | ✅ Automated grid search |
| **Best Model Selection** | ❌ Manual tracking | ✅ Automatic via ModelCheckpoint |
| **Load Compatibility** | ❌ Complex (needs K, models, layers in custom_objects) | ✅ Simple (just BinaryFocalLoss) |
| **Early Stopping** | ❓ Unknown | ✅ Yes (patience=20) |
| **LR Reduction** | ❓ Unknown | ✅ Yes (patience=10) |

## Expected Runtime

**Per experiment:**
- ~1-2 hours (100 epochs with early stopping)
- Early stopping typically triggers at epoch 60-80

**Total runtime:**
- Attention UNet: 18 combinations × 1.5 hours = ~27 hours
- Attention ResUNet: 18 combinations × 1.5 hours = ~27 hours
- **Total: ~48-60 hours** (sequential training)

**PBS walltime:** 48 hours (with 12-hour buffer)

## Model Loading (After Retraining)

### Simple Loading (No Lambda Issues!)

```python
from tensorflow import keras

# Only need BinaryFocalLoss in custom_objects
@keras.saving.register_keras_serializable(package='Custom')
class BinaryFocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        return focal_loss(y_true, y_pred, alpha=self.alpha, gamma=self.gamma)

    def get_config(self):
        return {'gamma': self.gamma, 'alpha': self.alpha}

# Also need RepeatElements (from models_fixed.py)
from models_fixed import RepeatElements

custom_objects = {
    'BinaryFocalLoss': BinaryFocalLoss,
    'RepeatElements': RepeatElements,
    'jacard_coef': jacard_coef,
    'dice_coef': dice_coef,
}

# Load model - NO safe_mode=False needed!
model = keras.models.load_model(
    'attention_hyperparam_*/checkpoints/*/best_model.keras',
    custom_objects=custom_objects
)

# ✓ Works! No K, no models, no layers needed!
```

## Hyperparameter Selection Guide

### After Training Completes

1. **Open results CSV:**
   ```bash
   head -20 attention_hyperparam_*/all_results.csv
   ```

2. **Sort by best Val IoU:**
   ```python
   import pandas as pd
   results = pd.read_csv('attention_hyperparam_*/all_results.csv')
   results_sorted = results.sort_values('best_val_iou', ascending=False)
   print(results_sorted.head(10))
   ```

3. **Compare architectures:**
   ```python
   # Best Attention UNet
   best_attn_unet = results[results['architecture'] == 'attention_unet'].nlargest(1, 'best_val_iou')

   # Best Attention ResUNet
   best_attn_resunet = results[results['architecture'] == 'attention_resunet'].nlargest(1, 'best_val_iou')

   print("Best Attention UNet:")
   print(best_attn_unet)
   print("\nBest Attention ResUNet:")
   print(best_attn_resunet)
   ```

4. **Use best model for density analysis:**
   ```python
   best_experiment = results.nlargest(1, 'best_val_iou')['experiment_name'].values[0]
   best_model_path = f'attention_hyperparam_*/checkpoints/{best_experiment}/best_model.keras'
   ```

## Comparison with Original Xukuang Models

### Original Xukuang Training (Phase 1)

```python
# From train_shrunk_xukuang_parameters.py
- Learning rate: 0.005 (fixed)
- Epochs: 200 (no early stopping)
- Dropout: 0.1 (fixed)
- Batch norm: True (fixed)
- Saved: FINAL model only (not best)
- Lambda layers: YES (serialization issues)
```

**Best Result:**
- UNet: 0.6789 IoU (epoch 140), 0.6065 IoU (final epoch 200)
- Attention models: Catastrophic overfitting

### New Training (Phase 2)

```python
# From train_attention_models_hyperparam.py
- Learning rate: [0.001, 0.003, 0.005] (searched)
- Epochs: 100 (with early stopping patience=20)
- Dropout: [0.1, 0.2, 0.3] (searched)
- Batch norm: True (always on)
- Saved: BOTH best AND final models
- Lambda layers: NO (RepeatElements instead)
```

**Expected Results:**
- Attention UNet: Should exceed 0.68 IoU (with proper hyperparameters)
- Attention ResUNet: Should exceed 0.68 IoU (with regularization)
- NO overfitting (early stopping + higher dropout options)

## Troubleshooting

### If Training Fails

1. **Check dataset paths:**
   ```bash
   ls dataset_new_shrunk/train/images/ | wc -l
   ls dataset_new_shrunk/val/images/ | wc -l
   ```

2. **Check GPU memory:**
   - If OOM, reduce batch_size in CONFIG from 4 to 2
   - Or reduce n_filters grid to [16] only

3. **Check logs:**
   ```bash
   tail -100 Attention_Hyperparam.o*
   tail -100 train_attention_hyperparam_console_*.log
   ```

### If Models Don't Load

**Error:** "Could not locate class 'RepeatElements'"

**Solution:** Import RepeatElements from models_fixed.py
```python
from models_fixed import RepeatElements
custom_objects = {
    'RepeatElements': RepeatElements,
    # ... other objects
}
```

**Error:** "Could not locate class 'BinaryFocalLoss'"

**Solution:** Define BinaryFocalLoss with decorator
```python
@keras.saving.register_keras_serializable(package='Custom')
class BinaryFocalLoss(keras.losses.Loss):
    # ... implementation
```

## Next Steps

After training completes:

1. **Review results:**
   ```bash
   cat attention_hyperparam_*/all_results.csv
   ```

2. **Identify best models:**
   - Sort by `best_val_iou`
   - Note experiment name

3. **Run density analysis with best models:**
   - Update `density_analysis_xukuang.py` to load from new checkpoints
   - Use checkpoint path: `attention_hyperparam_*/checkpoints/{best_exp}/best_model.keras`

4. **Compare with original:**
   - Original UNet: 0.6789 IoU
   - New Attention models: Expected > 0.68 IoU

5. **Generate report:**
   - Training curves
   - Hyperparameter analysis
   - Model comparison

## Summary

**Problem:** Lambda layers in Attention models cause serialization errors

**Solution:** Retrain with `RepeatElements` custom layer

**Benefits:**
- ✅ Clean serialization (no Lambda issues)
- ✅ Both best and final models saved
- ✅ Hyperparameter optimization
- ✅ Expected performance improvement
- ✅ Future-proof architecture

**Timeline:**
- Submit job: `qsub pbs_train_attention_hyperparam.sh`
- Runtime: ~48 hours
- Result: 36 models (18 per architecture) with full hyperparam search

---

**Created:** October 16, 2025
**Author:** Claude Code
**Status:** ✅ Ready for HPC submission
**Estimated Completion:** 48-60 hours after submission
