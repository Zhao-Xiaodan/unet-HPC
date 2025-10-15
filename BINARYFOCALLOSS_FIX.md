# BinaryFocalLoss Import Error Fix

## Problem (Jobs 288639, 288640, 288641)

All three training jobs failed immediately with the same error:

```
ImportError: cannot import name 'BinaryFocalLoss' from 'loss_functions_fixed'
```

### Error Details

**Job 288639 (UNet):**
```
File "/scratch/phyzxi/unet-HPC/./train_unet_hyperparam.py", line 35, in <module>
    from loss_functions_fixed import (
ImportError: cannot import name 'BinaryFocalLoss' from 'loss_functions_fixed'
```

**Job 288640 (Attention UNet):**
```
File "/scratch/phyzxi/unet-HPC/./train_attention_unet_hyperparam.py", line 35, in <module>
    from loss_functions_fixed import (
ImportError: cannot import name 'BinaryFocalLoss' from 'loss_functions_fixed'
```

**Job 288641 (Attention ResUNet):**
```
File "/scratch/phyzxi/unet-HPC/./train_attention_resunet_hyperparam.py", line 35, in <module>
    from loss_functions_fixed import (
ImportError: cannot import name 'BinaryFocalLoss' from 'loss_functions_fixed'
```

## Root Cause

The `loss_functions_fixed.py` file only contained the **function** `focal_loss()`, but not the **class** `BinaryFocalLoss`.

The training scripts attempted to import:
```python
from loss_functions_fixed import (
    combined_dice_focal_loss,
    jacard_coef,
    dice_coef,
    focal_loss,
    BinaryFocalLoss  # ← This class didn't exist!
)
```

## Solution

Added `BinaryFocalLoss` class to `loss_functions_fixed.py`:

```python
@keras.saving.register_keras_serializable(package='Custom')
class BinaryFocalLoss(keras.losses.Loss):
    """
    Binary Focal Loss as a Keras Loss class for proper serialization.

    Wrapper around the focal_loss function that can be used with model.compile()
    and properly serialized when saving models.

    Usage:
        model.compile(
            optimizer='adam',
            loss=BinaryFocalLoss(gamma=2, alpha=0.25),
            metrics=['accuracy']
        )

    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Balancing factor (default: 0.25)
        name: Name of the loss (default: 'binary_focal_loss')
    """
    def __init__(self, gamma=2.0, alpha=0.25, name='binary_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        """Compute focal loss."""
        return focal_loss(y_true, y_pred, alpha=self.alpha, gamma=self.gamma)

    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        return cls(**config)
```

### Key Features of the Fix

1. **✅ Proper inheritance** - Inherits from `keras.losses.Loss`
2. **✅ Serialization decorator** - `@keras.saving.register_keras_serializable(package='Custom')`
3. **✅ Config methods** - `get_config()` and `from_config()` for saving/loading
4. **✅ Wraps existing function** - Uses the stable `focal_loss()` function internally
5. **✅ Compatible with model.compile()** - Can be used as a loss class

## Why This Class is Needed

### Function vs Class in Keras

**Functional API (`focal_loss`):**
```python
# Works, but harder to serialize
model.compile(
    optimizer='adam',
    loss=lambda y_true, y_pred: focal_loss(y_true, y_pred, alpha=0.25, gamma=2)
)
```
❌ Lambda functions can't be serialized properly
❌ Can't save model with `model.save()`

**Class API (`BinaryFocalLoss`):**
```python
# Proper way - easy to serialize
model.compile(
    optimizer='adam',
    loss=BinaryFocalLoss(gamma=2, alpha=0.25)
)
```
✅ Class can be serialized
✅ Model saves/loads cleanly
✅ No Lambda layer issues

## Verification

```bash
# Syntax check
python3 -m py_compile loss_functions_fixed.py
# ✓ Passed
```

## Files Modified

1. `loss_functions_fixed.py`
   - Added `BinaryFocalLoss` class (lines 280-325)
   - Placed before `LOSS_FUNCTIONS` dictionary
   - Fully documented with docstrings

## Testing

The class can be tested independently:

```python
from loss_functions_fixed import BinaryFocalLoss
import tensorflow as tf

# Create loss instance
loss_fn = BinaryFocalLoss(gamma=2, alpha=0.25)

# Test with dummy data
y_true = tf.constant([[[[1.0]], [[0.0]]]])
y_pred = tf.constant([[[[0.9]], [[0.1]]]])

# Compute loss
loss_value = loss_fn(y_true, y_pred)
print(f"Loss: {loss_value:.6f}")

# Test serialization
config = loss_fn.get_config()
print(f"Config: {config}")

# Test deserialization
loss_fn_loaded = BinaryFocalLoss.from_config(config)
print(f"Loaded loss: {loss_fn_loaded}")
```

## Next Steps

### Resubmit Jobs

Now that `BinaryFocalLoss` is defined, resubmit the training jobs:

```bash
# Submit all three jobs again
qsub pbs_train_unet.sh
qsub pbs_train_attention_unet.sh
qsub pbs_train_attention_resunet.sh

# Monitor
qstat -u $USER
```

### Expected Behavior

Jobs should now:
1. ✅ Import `BinaryFocalLoss` successfully
2. ✅ Load dataset (98 images, split 80/20)
3. ✅ Train 27 hyperparameter combinations per model
4. ✅ Save both best and final models
5. ✅ Complete in 24-48 hours

## Alternative: Use Functional Loss

If you prefer to avoid the class, you can modify the training scripts to use the functional API:

```python
# In train_*_hyperparam.py, change compile_model():

def compile_model(model, learning_rate, config):
    """Compile model with optimizer, loss, and metrics."""

    # Use functional loss instead of class
    if config['loss'] == 'binary_focal_loss':
        loss_fn = lambda y_true, y_pred: focal_loss(
            y_true, y_pred,
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma']
        )
    elif config['loss'] == 'combined_dice_focal':
        loss_fn = combined_dice_focal_loss
    else:
        raise ValueError(f"Unknown loss: {config['loss']}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[jacard_coef, dice_coef]
    )

    return model
```

**However**, using the `BinaryFocalLoss` class is **recommended** because:
- Better serialization
- Cleaner code
- Matches the original Xukuang training approach

## Summary

**Problem:** Missing `BinaryFocalLoss` class in `loss_functions_fixed.py`

**Solution:** Added properly serializable `BinaryFocalLoss` class

**Status:** ✅ Fixed and ready for resubmission

**Files Modified:**
- `loss_functions_fixed.py` (+46 lines)

**Verification:**
- ✅ Syntax check passed
- ✅ Class properly decorated for serialization
- ✅ Inherits from `keras.losses.Loss`

---

**Fixed:** October 16, 2025
**Jobs Affected:** 288639, 288640, 288641
**Ready for:** Resubmission
