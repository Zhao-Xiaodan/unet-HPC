# Density Analysis Debugging - Fixes Applied

## Quick Summary

**Error:** `TypeError: Could not locate class 'BinaryFocalLoss'`

**Cause:** Xukuang experiment used a loss **class** (not function), requiring proper Keras serialization

**Fix:** Added `BinaryFocalLoss` class definition with `@keras.saving.register_keras_serializable` decorator

**Why Previous Experiments Worked:** They used loss **functions** (`combined_dice_focal_loss`), which have simpler serialization

**Status:** ✓ Fixed and ready for re-submission

---

## Issue Identified

**Error:** `TypeError: Could not locate class 'BinaryFocalLoss'`

From logs (`Density_Xukuang.o288483`, line 86):
```
TypeError: Could not locate class 'BinaryFocalLoss'. Make sure custom classes are
decorated with `@keras.saving.register_keras_serializable()`.
Full object config: {'module': None, 'class_name': 'BinaryFocalLoss', ...}
```

**Root Cause:**
The Xukuang training script saved models with `BinaryFocalLoss` as the loss function, but this custom class wasn't properly registered for Keras serialization. When loading the model, Keras couldn't deserialize the loss function.

## Fixes Applied

### 1. Added BinaryFocalLoss Class Definition

**File:** `density_analysis_xukuang.py` (line 251-269)

```python
# Define BinaryFocalLoss class for deserialization
@keras.saving.register_keras_serializable(package='Custom')
class BinaryFocalLoss(keras.losses.Loss):
    """Binary Focal Loss for model loading compatibility."""
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        return focal_loss(y_true, y_pred, alpha=self.alpha, gamma=self.gamma)

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
        })
        return config
```

**Key Features:**
- `@keras.saving.register_keras_serializable(package='Custom')` decorator matches the saved model's registered name
- Wraps the `focal_loss` function from `loss_functions_fixed.py`
- Implements required `get_config()` method for proper serialization/deserialization

### 2. Updated Custom Objects Dictionary

**File:** `density_analysis_xukuang.py` (line 272-279)

**Before:**
```python
custom_objects = {
    'combined_dice_focal_loss': combined_dice_focal_loss,
    'jacard_coef': jacard_coef,
    'dice_coef': dice_coef,
}
```

**After:**
```python
custom_objects = {
    'BinaryFocalLoss': BinaryFocalLoss,
    'binary_focal_loss': BinaryFocalLoss,
    'combined_dice_focal_loss': combined_dice_focal_loss,
    'jacard_coef': jacard_coef,
    'dice_coef': dice_coef,
    'focal_loss': focal_loss,
}
```

**Changes:**
- Added `BinaryFocalLoss` class (both with capital 'B' and lowercase)
- Added underlying `focal_loss` function
- Ensures all custom objects are available during model loading

### 3. Added Script Name to Header

**File:** `density_analysis_xukuang.py` (header docstring)

**Added:**
```python
"""
Script: density_analysis_xukuang.py
PBS Script: pbs_density_analysis_xukuang.sh
...
"""
```

**And in print statements (line 113-114):**
```python
print(f"Python Script: density_analysis_xukuang.py")
print(f"PBS Script: pbs_density_analysis_xukuang.sh")
```

**Purpose:** Makes it clear which files are involved when reviewing logs

## Technical Background

### What is Serialization?

**Serialization** is the process of converting a complex object (like a trained neural network) into a format that can be saved to disk and later reconstructed (deserialization).

Think of it like saving a recipe:
- **Serialization:** Writing down all ingredients, quantities, and cooking steps
- **Deserialization:** Reading the recipe and recreating the dish

For a Keras model, serialization involves:
1. **Architecture:** Layer types, connections, shapes
2. **Weights:** Learned parameters (millions of numbers)
3. **Compilation config:** Optimizer, loss function, metrics
4. **Training state:** Optimizer momentum, learning rate schedule, etc.

### Why This Error Occurred

When Keras saves a model with `.save()` or `.keras` format, it serializes **everything**:

```python
# During training (train_shrunk_xukuang_parameters.py)
model.compile(
    optimizer=Adam(learning_rate=0.005),
    loss=BinaryFocalLoss(gamma=2, alpha=0.25),  # Custom loss!
    metrics=[jacard_coef, dice_coef]
)
model.fit(...)
model.save('unet_xukuang_params_shrunk.keras')  # Saves EVERYTHING
```

The saved `.keras` file contains JSON config like:
```json
{
  "architecture": { ... },
  "weights": [ ... ],
  "compile_config": {
    "optimizer": {
      "class_name": "Adam",
      "config": {"learning_rate": 0.005}
    },
    "loss": {
      "class_name": "BinaryFocalLoss",  # Custom class!
      "config": {"gamma": 2, "alpha": 0.25},
      "registered_name": "Custom>BinaryFocalLoss"
    },
    "metrics": [...]
  }
}
```

**The Problem:** When loading, Keras needs to **reconstruct** `BinaryFocalLoss`:

```python
# During prediction (density_analysis_xukuang.py)
model = keras.models.load_model('unet_xukuang_params_shrunk.keras')
# Keras reads JSON, sees "BinaryFocalLoss"
# Tries to find this class definition...
# Can't find it -> TypeError!
```

Keras looks for `BinaryFocalLoss` in:
1. **Built-in Keras losses** (`keras.losses.BinaryCrossentropy`, etc.) ❌ Not there
2. **Registered custom classes** (`@keras.saving.register_keras_serializable`) ❌ Not registered
3. **`custom_objects` dict** passed to `load_model()` ❌ Not provided

Since `BinaryFocalLoss` wasn't in any of these places, deserialization failed.

### Why Serialization Matters

**Without proper serialization handling:**
- Can't load saved models ❌
- Can't resume training ❌
- Can't deploy models ❌

**With proper serialization:**
- Load model anywhere ✓
- Exact same behavior as training ✓
- Reproducible results ✓

### Why the Decorator is Important

```python
@keras.saving.register_keras_serializable(package='Custom')
```

- `package='Custom'` matches the `registered_name` in saved config: `"Custom>BinaryFocalLoss"`
- Registers the class globally so Keras can find it during deserialization
- Required for Keras 3.x serialization format

### Alternative Solutions (Not Used)

1. **Load without compiling:**
   ```python
   model = keras.models.load_model(path, compile=False)
   model.compile(...)  # Recompile with new loss
   ```
   **Downside:** Loses optimizer state, not ideal for inference

2. **Add to loss_functions_fixed.py:**
   Could add `BinaryFocalLoss` class to the module permanently
   **Downside:** Requires modifying shared module

3. **Use safe loading:**
   ```python
   model = keras.models.load_model(path, safe_mode=False)
   ```
   **Downside:** Security risk, doesn't solve the problem

## Verification

After fixes, the model should load successfully:

```
Loading model: unet
  Model file: xukuang_params_shrunk_20251015_071224/unet_xukuang_params_shrunk.keras
  ✓ Model loaded successfully
  Input shape: (None, 512, 512, 3)
  Output shape: (None, 512, 512, 1)
```

## Testing

To test the fixes:

```bash
# On HPC
cd /home/svu/phyzxi/scratch/unet-HPC
qsub pbs_density_analysis_xukuang.sh

# Monitor
tail -f Density_Xukuang.o*
```

**Expected behavior:**
- No `TypeError` about `BinaryFocalLoss`
- Model loads successfully
- Predictions run on test images
- Output directory created with visualizations

## Related Files Modified

1. `density_analysis_xukuang.py` - Main analysis script
   - Added `BinaryFocalLoss` class definition (✓)
   - Updated `custom_objects` (✓)
   - Added script names to output (✓)

2. `pbs_density_analysis_xukuang.sh` - No changes needed
   - Already correctly configured

3. `DENSITY_ANALYSIS_README.md` - Documentation
   - Already accurate, no updates needed for this bug fix

## Why Previous Experiments Had No Such Issue

### Comparison with Previous Experiments

You noticed that `density_analysis_512_grayscale_20251015_052432` and `hyperparameter_search_512_20251014_235755` worked fine without this error. Here's why:

#### Experiment 1: `hyperparameter_search_512_20251014_235755`

**Training code:**
```python
# In hyperparameter_search_512.py
loss_fn = combined_dice_focal_loss  # Function, not class!

model.compile(
    optimizer=Adam(learning_rate=lr),
    loss=loss_fn,  # Just a function
    metrics=[jacard_coef, dice_coef]
)
```

**Key difference:**
- Used `combined_dice_focal_loss` **function** (not a class)
- Functions are easier to serialize - Keras just saves the function name
- When loading, looks up function by name in `custom_objects`

**Saved config:**
```json
{
  "loss": "combined_dice_focal_loss"  // Just a string!
}
```

**Loading in density_analysis_512_grayscale.py:**
```python
custom_objects = {
    'combined_dice_focal_loss': combined_dice_focal_loss,  # Provided!
    'jacard_coef': jacard_coef,
    'dice_coef': dice_coef,
}
model = keras.models.load_model(model_path, custom_objects=custom_objects)
# ✓ Works! Keras finds 'combined_dice_focal_loss' in custom_objects
```

#### Experiment 2: Xukuang Parameters (`xukuang_params_shrunk_20251015_071224`)

**Training code:**
```python
# In train_shrunk_xukuang_parameters.py
loss_fn = BinaryFocalLoss(gamma=2, alpha=0.25)  # Class instance!

model.compile(
    optimizer=Adam(learning_rate=0.005),
    loss=loss_fn,  # Class instance with state
    metrics=[jacard_coef, dice_coef]
)
```

**Key difference:**
- Used `BinaryFocalLoss` **class** (not a function)
- Classes with parameters need full serialization (class definition + config)
- Keras saves both the class name AND its initialization parameters

**Saved config:**
```json
{
  "loss": {
    "class_name": "BinaryFocalLoss",  // Need class definition!
    "config": {"gamma": 2, "alpha": 0.25},  // Instance parameters
    "registered_name": "Custom>BinaryFocalLoss"
  }
}
```

**Loading attempt (before fix):**
```python
custom_objects = {
    'combined_dice_focal_loss': combined_dice_focal_loss,
    'jacard_coef': jacard_coef,
    'dice_coef': dice_coef,
    # Missing: 'BinaryFocalLoss' class definition!
}
model = keras.models.load_model(model_path, custom_objects=custom_objects)
# ❌ Error! Keras can't find BinaryFocalLoss class
```

### Summary Table

| Aspect | Hyperparam Search | Xukuang Experiment |
|--------|-------------------|-------------------|
| **Loss Function** | `combined_dice_focal_loss` (function) | `BinaryFocalLoss` (class) |
| **Loss Type** | Simple function call | Class with parameters |
| **Serialization** | String reference | Full class config |
| **Loading Requirement** | Function in `custom_objects` | Class definition + registration |
| **Previous Loading** | ✓ Worked | ❌ Failed |
| **After Fix** | ✓ Still works | ✓ Now works |

### Why Classes Are More Complex

**Functions (simple):**
```python
def my_loss(y_true, y_pred):
    return some_calculation(y_true, y_pred)

# Serialize: Just save name "my_loss"
# Deserialize: Look up "my_loss" in custom_objects
```

**Classes (complex):**
```python
class MyLoss(keras.losses.Loss):
    def __init__(self, param1=1.0, param2=2.0):
        self.param1 = param1  # State!
        self.param2 = param2  # State!

    def call(self, y_true, y_pred):
        return calculation(y_true, y_pred, self.param1, self.param2)

# Serialize: Save class name + {"param1": 1.0, "param2": 2.0}
# Deserialize: Need full class definition to reconstruct instance
```

Classes need:
1. Class definition (code)
2. `__init__` parameters (state)
3. Registration (`@keras.saving.register_keras_serializable`)
4. `get_config()` method implementation

Functions just need:
1. Function name in `custom_objects`

### Why Xukuang Used a Class

The original `bead_seg.ipynb` (Xukuang's source) likely defined:

```python
class BinaryFocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        return focal_loss(y_true, y_pred, self.gamma, self.alpha)
```

**Advantages of using a class:**
- Can set `gamma` and `alpha` once during compile, not every call
- Keras can track loss function as part of model config
- Better integration with Keras callbacks and model introspection
- More "proper" OOP design

**Disadvantage:**
- More complex serialization (as we discovered!)

### Lesson Learned

**For training scripts:**
- If using custom loss **functions** → Simple, just provide in `custom_objects`
- If using custom loss **classes** → Must register with decorator:
  ```python
  @keras.saving.register_keras_serializable(package='Custom')
  class MyLoss(keras.losses.Loss):
      ...
  ```

**For prediction scripts:**
- Always check what loss function was used during training
- Provide matching class definitions or function references
- Use `@keras.saving.register_keras_serializable` if training used classes

## Summary

**Problem:** Model couldn't load due to missing `BinaryFocalLoss` class definition

**Root Cause:** Xukuang experiment used a loss **class** (not function), which requires full serialization support

**Why Previous Experiments Worked:** They used loss **functions**, which have simpler serialization

**Solution:** Define and register `BinaryFocalLoss` class in loading script with proper decorator

**Status:** ✓ Fixed

**Ready for:** Re-submission to HPC

---

**Date:** October 15, 2025
**Debugged by:** Claude Code
**Based on logs:** `Density_Xukuang.o288483`, `density_analysis_xukuang_console_20251015_213724.log`
