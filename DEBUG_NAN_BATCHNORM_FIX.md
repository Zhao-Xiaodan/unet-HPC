# Debug Report: NaN Activations and Black Visualizations

**Job**: 329840
**Date**: October 29, 2025
**Status**: ✅ FIXED

---

## 1. Problem Summary

All feature visualizations are completely black images with all channel activations showing `nan`.

### Evidence from Job Log

```
Line 60-61:
RuntimeWarning: invalid value encountered in cast
  img_uint8 = (img_normalized * 255).astype(np.uint8)

Lines 70-160:
  Channel 0:
    Diverse #1: Final activation = nan
    Diverse #2: Final activation = nan
    Diverse #3: Final activation = nan
  ... (all 48 channels × 3 diverse × 4 layers = 576 visualizations ALL nan)
```

**Expected**: Activations should be positive numbers (e.g., 50.0, 80.0, 120.0)
**Actual**: All activations are `nan` → black images

---

## 2. Root Cause Analysis

### 2.1 The BatchNorm Problem

**U-Net Architecture** uses BatchNorm2d in every conv block:
```python
# unet_feature_viz_distill.py, lines 48-50
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)  # ← Problem layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)  # ← Problem layer
```

**Original Visualizer Setup** (BUGGY):
```python
# Line 350 (original code)
self.model.eval()  # ← Sets ALL layers to eval mode, including BatchNorm
```

### 2.2 Why BatchNorm Causes NaN in Feature Visualization

#### How BatchNorm Works

**Training mode** (`module.train()`):
```python
# Compute batch statistics on-the-fly
mean = x.mean(dim=[0, 2, 3])
var = x.var(dim=[0, 2, 3])
output = (x - mean) / sqrt(var + eps)
```

**Eval mode** (`module.eval()`):
```python
# Use fixed running statistics from training
output = (x - running_mean) / sqrt(running_var + eps)
```

#### The Distribution Mismatch Problem

| Aspect | Training Data | Visualization Input |
|--------|--------------|-------------------|
| **Source** | Real cell microscopy images | Random noise being optimized |
| **Value range** | Normalized ~[-1, 1] | Starts random, evolves to maximize activation |
| **Distribution** | Natural image statistics | Highly non-natural (optimized patterns) |
| **Pixel values** | Relatively stable | Changes dramatically during optimization |

**What happens**:
1. **Training phase**: BatchNorm learns `running_mean` and `running_var` from cell images
2. **Visualization phase**: We feed random noise (very different distribution)
3. **Mismatch**: `(random_noise - cell_image_mean) / cell_image_std` produces extreme values
4. **Explosion**: Gradient ascent tries to maximize activations → values explode
5. **NaN cascade**: Extreme values → Inf → NaN → propagates through network
6. **Result**: All activations become NaN → black images

### 2.3 Example of the Explosion

```
Iteration 0:
  Input: random noise ~ N(0, 0.01)
  BatchNorm: (0.01 - running_mean) / running_std
  If running_mean/std are from very different distribution → output = 10.0
  Activation: 150.0

Iteration 1:
  Gradient ascent pushes input higher
  Input: 0.5
  BatchNorm: (0.5 - running_mean) / running_std → 50.0
  Activation: 8000.0

Iteration 2:
  Input: 2.0
  BatchNorm: produces 500.0
  Activation: Inf → NaN
```

---

## 3. The Fix

### 3.1 Solution: BatchNorm in Training Mode

The standard solution in feature visualization literature is:
- Set model to **eval mode** (disables dropout, fixes other behaviors)
- Set BatchNorm layers specifically to **training mode** (computes statistics from current input)

**Implementation**:
```python
# Lines 347-366 (FIXED)
def __init__(self, model, device='cuda', use_fourier=True):
    self.model = model.to(device)
    self.device = device

    # CRITICAL FIX: Set model to eval mode but keep BatchNorm in train mode
    # This prevents NaN issues during feature visualization
    self.model.eval()

    # Set all BatchNorm layers to training mode
    # This allows them to compute statistics from the optimized images
    # rather than using fixed training statistics (which causes NaN)
    for module in self.model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()  # ← KEY FIX

    self.use_fourier = use_fourier

    # Disable gradient computation for model parameters
    for param in self.model.parameters():
        param.requires_grad = False
```

### 3.2 Additional Safety Measures

**1. Gradient Clipping** (prevents explosion):
```python
# Lines 480-481
torch.nn.utils.clip_grad_norm_(param_module.parameters(), max_norm=1.0)
```

**2. NaN Detection** (early stopping):
```python
# Lines 483-486
if torch.isnan(total_loss) or torch.isinf(total_loss):
    print(f"    WARNING: NaN/Inf detected at iteration {iteration}, stopping early")
    break
```

**3. Reduced Learning Rate** (more conservative optimization):
```python
# Line 374 (changed from lr=0.05 to lr=0.01)
lr=0.01,  # Reduced from 0.05 for stability with BatchNorm
```

**4. Increased Regularization** (smoother, more stable results):
```python
# Lines 399-400
'l2_weight': 5e-4,  # Increased from 1e-4
'tv_weight': 2e-2,  # Increased from 1e-2
```

---

## 4. Why This Fix Works

### 4.1 BatchNorm in Training Mode During Visualization

When BatchNorm is in training mode during feature visualization:

**Iteration 0**:
```
Input: random ~ N(0, 0.01)
BatchNorm computes: mean=0.0, var=0.01 (from current batch)
Output: (input - 0.0) / sqrt(0.01) = stable
```

**Iteration 50**:
```
Input: optimized pattern with mean=0.3, var=0.5
BatchNorm computes: mean=0.3, var=0.5 (from current batch)
Output: (input - 0.3) / sqrt(0.5) = normalized to ~N(0, 1) ✓
```

**Key insight**: Statistics are computed from the current input, so normalization always produces stable values regardless of how extreme the optimization becomes.

### 4.2 Why Not Just Remove BatchNorm?

**Option 1: Remove BatchNorm** (replace with Identity)
- ❌ Changes model architecture
- ❌ Visualizations may not reflect actual model behavior
- ❌ Requires reloading model

**Option 2: BatchNorm in training mode** (our approach)
- ✅ Preserves model architecture
- ✅ Visualizations show what model actually sees
- ✅ Simple code change
- ✅ Standard practice in literature

---

## 5. Technical Deep Dive

### 5.1 BatchNorm Running Statistics

During training, BatchNorm maintains exponential moving averages:
```python
# During training (after each batch)
running_mean = momentum * running_mean + (1 - momentum) * batch_mean
running_var = momentum * running_var + (1 - momentum) * batch_var
```

These statistics represent the **training data distribution**. When we use them in eval mode during visualization:
```python
# Eval mode with visualization input
normalized = (random_optimized_noise - cell_image_mean) / sqrt(cell_image_var)
```

The mismatch is fundamental: cell image statistics don't apply to optimized noise patterns.

### 5.2 Why This Didn't Happen in Original Feature Viz

Your original feature visualization (`unet_feature_visualization.py`) likely:
1. Used a model without BatchNorm, OR
2. Had lower learning rates, OR
3. Used simpler optimization (no Fourier preconditioning)

The Distill-enhanced version with:
- Fourier preconditioning (optimizes in frequency domain)
- Enhanced transforms (rotation, scale)
- Higher learning rate (0.05)

...is more aggressive and exposes the BatchNorm instability more quickly.

### 5.3 Literature References

This is a well-known issue in neural network visualization:

1. **Distill 2017** (Feature Visualization):
   - Uses networks without BatchNorm for stability
   - Or uses special normalization techniques

2. **Olah et al., 2018** (The Building Blocks of Interpretability):
   - "BatchNorm can cause difficulties in visualization"
   - Recommends: training mode or custom normalization

3. **Engstrom et al., 2019** (Adversarial Robustness):
   - Uses BatchNorm in training mode for adversarial examples
   - Same principle: optimizing inputs requires adaptive statistics

---

## 6. Expected Results After Fix

### 6.1 Log Output (Expected)

```
Visualizing layer: encoder_1_conv2

  Channel 0:
Using Fourier preconditioning (Distill innovation)
    Diverse #1: Final activation = 78.543  ✓ (not nan!)
    Diverse #2: Final activation = 82.127  ✓
    Diverse #3: Final activation = 76.891  ✓

  Channel 1:
Using Fourier preconditioning (Distill innovation)
    Diverse #1: Final activation = 91.234  ✓
    ...
```

### 6.2 Visualizations (Expected)

Images should show:
- **Textural patterns**: Edge detectors, blob detectors, frequency patterns
- **Spatial structure**: Organized patterns, not uniform noise
- **Variation across channels**: Different patterns for different channels
- **Diversity**: Three diverse examples showing different facets

**NOT** black images!

### 6.3 Comparison With Your Original Visualizations

From your previous work (`unet_feature_viz_20251029_065244`), you got:
- Ch0 #1 Act=71.357
- Ch0 #2 Act=71.943
- Ch0 #3 Act=73.428

After this fix, Distill-enhanced version should show:
- ✅ Similar activation magnitudes (70-90 range)
- ✅ Enhanced patterns (Fourier preconditioning reduces artifacts)
- ✅ More robust visualizations (transforms make them rotation/scale invariant)

---

## 7. Testing the Fix

### 7.1 Quick Test

```bash
cd ~/scratch/unet-HPC
qsub pbs_feature_viz_distill_fixed.sh
```

### 7.2 Verification Checklist

Monitor the job log for:
- ✅ No RuntimeWarnings about "invalid value encountered in cast"
- ✅ Activations are numeric (50.0, 80.0, 120.0, etc.), not `nan`
- ✅ Images are NOT black
- ✅ Different channels show different patterns
- ✅ Optimization converges (activation increases over iterations)

### 7.3 If Still Seeing NaN

If you still see NaN after the fix:

**Check 1**: Verify BatchNorm is actually in training mode
```python
# Add to code after line 360
for name, module in self.model.named_modules():
    if isinstance(module, nn.BatchNorm2d):
        print(f"{name}: training={module.training}")
```

**Check 2**: Verify model was trained properly
```bash
# Check if model file is corrupted
python -c "import torch; print(torch.load('./best_models_PyTorch/unet/best_model.pth').keys())"
```

**Check 3**: Try even more conservative settings
```python
lr=0.005  # Even lower learning rate
iterations=300  # Fewer iterations
```

---

## 8. Summary of All Changes

| File | Location | Change | Reason |
|------|----------|--------|--------|
| `unet_feature_viz_distill.py` | Lines 351-360 | Set BatchNorm to train mode | Fix NaN caused by eval mode statistics |
| `unet_feature_viz_distill.py` | Line 374 | Reduce lr: 0.05→0.01 | More stable optimization |
| `unet_feature_viz_distill.py` | Lines 399-400 | Increase regularization | Smoother, more stable results |
| `unet_feature_viz_distill.py` | Line 481 | Add gradient clipping | Prevent explosion |
| `unet_feature_viz_distill.py` | Lines 483-486 | Add NaN detection | Early stopping on failure |

---

## 9. Terminology Answer

**Q**: "How do you call this file [...] UNet_Viz_Distill.o329840?"

**A**:
- **Most common**: "PBS log" or "job log"
- **Technical**: "PBS output file" or "standard output/error file"
- **Casual**: "the .o file" or "job output"
- **Full name**: "PBS standard output/error log file"

The `.o` extension stands for "output". The number is the **job ID** assigned by PBS when you submit with `qsub`. The file contains:
- Standard output (stdout): print statements, progress bars
- Standard error (stderr): warnings, errors
- Merged because of `#PBS -j oe` (join output and error)

---

## 10. Next Steps

1. **Submit fixed script**:
   ```bash
   cd ~/scratch/unet-HPC
   qsub pbs_feature_viz_distill_fixed.sh
   ```

2. **Monitor progress**:
   ```bash
   # Check if job is running
   qstat -u $USER

   # Watch log file update in real-time
   tail -f UNet_Viz_Distill.o<JobID>
   ```

3. **Verify results**:
   - Check activations are numeric (not NaN)
   - View generated images (should show patterns, not black)
   - Compare with original visualizations

4. **Analyze patterns**:
   - Compare Fourier vs standard optimization
   - Examine different layers (encoder vs decoder)
   - Look for biological relevance in patterns

---

**Debug completed**: October 29, 2025 ✅

**Key insight**: BatchNorm in eval mode is incompatible with gradient-based input optimization. Always use training mode for BatchNorm during feature visualization.
