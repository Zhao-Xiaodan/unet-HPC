# Bug Fix: PyTorch Attention Gate Channel Mismatch

**Date:** October 21, 2025
**Jobs Affected:**
- PyTorch_NoAug_Comparison.o302535
- PyTorch_WithAug_Comparison.o302537
- PyTorch_AdaptiveLoss_Comparison.o302538

**Status:** ✅ Fixed

---

## Problem

All three PyTorch comparison training jobs crashed when starting **Attention UNet** training with:

```python
RuntimeError: Given groups=1, weight of size [64, 256, 1, 1],
expected input[4, 128, 64, 64] to have 256 channels, but got 128 channels instead
```

**Error Location:** `AttentionGate.forward()` at line `g1 = self.W_g(g)`

**Result:** Only UNet training completed successfully (~27 configs). Attention UNet and Attention ResUNet failed immediately.

---

## Root Cause

### Incorrect Attention Gate Channel Configuration

**The Bug:**
```python
# WRONG - Used bottleneck output channels as F_g
self.att4 = AttentionGate(F_g=n_filters * 16, F_l=n_filters * 8, F_int=n_filters * 4)
#                              ^^^^^^^^^^^^^^
#                              This is WRONG!
```

**Why it's wrong:**

The attention gate `F_g` parameter should match the **gating signal channels**, which is the output of the **upsampling operation**, NOT the bottleneck:

```python
# In forward():
b = self.bottleneck(...)         # Output: n_filters * 16 channels
d4 = self.up4(b)                 # Output: n_filters * 8 channels (upsampled!)
e4_att = self.att4(g=d4, x=e4)   # d4 has n_filters * 8, not n_filters * 16!
```

**The upsampling reduces channels:**
```python
self.up4 = nn.ConvTranspose2d(
    n_filters * 16,    # Input channels (from bottleneck)
    n_filters * 8,     # Output channels (upsampled)
    2, stride=2
)
```

So `d4` has `n_filters * 8` channels, not `n_filters * 16`.

---

## Fix Applied

### Corrected Attention Gate Initialization

**For AttentionUNet and AttentionResUNet:**

```python
# OLD (WRONG):
self.att4 = AttentionGate(F_g=n_filters * 16, F_l=n_filters * 8, F_int=n_filters * 4)
self.att3 = AttentionGate(F_g=n_filters * 8,  F_l=n_filters * 4, F_int=n_filters * 2)
self.att2 = AttentionGate(F_g=n_filters * 4,  F_l=n_filters * 2, F_int=n_filters)
self.att1 = AttentionGate(F_g=n_filters * 2,  F_l=n_filters,     F_int=n_filters // 2)

# NEW (CORRECT):
self.att4 = AttentionGate(F_g=n_filters * 8,  F_l=n_filters * 8, F_int=n_filters * 4)
self.att3 = AttentionGate(F_g=n_filters * 4,  F_l=n_filters * 4, F_int=n_filters * 2)
self.att2 = AttentionGate(F_g=n_filters * 2,  F_l=n_filters * 2, F_int=n_filters)
self.att1 = AttentionGate(F_g=n_filters,      F_l=n_filters,     F_int=n_filters // 2)
```

**Key Change:** `F_g` now matches the **upsampled decoder channels** at each level, not the pre-upsampling channels.

---

## Channel Flow Diagram

```
Bottleneck (n_filters * 16)
    ↓
up4: ConvTranspose2d(16 → 8 filters)
    ↓
d4 (n_filters * 8) ──┐
                     │
Encoder e4 (n_filters * 8) ──┐
                              │
                         AttentionGate
                         F_g = n_filters * 8  ← gating (d4)
                         F_l = n_filters * 8  ← skip (e4)
                              │
                         e4_att ──→ Concatenate with d4
                                         │
                                    dec4 (16 → 8 filters)
                                         ↓
                                    (continue decoder...)
```

---

## Files Modified

✅ `train_pytorch_comparison_no_aug.py`
- AttentionUNet class (line ~395)
- AttentionResUNet class (line ~472)

✅ `train_pytorch_comparison_with_aug.py`
- AttentionUNet class
- AttentionResUNet class

✅ `train_pytorch_comparison_adaptive_loss.py`
- AttentionUNet class
- AttentionResUNet class

**All instances fixed:** 6 total (2 classes × 3 files)

---

## Why This Error Occurred

### Misunderstanding of Attention Gate Parameters

The original implementation incorrectly assumed:
- `F_g` = bottleneck output channels
- But actually: `F_g` = **gating signal** channels (after upsampling)

### Reference Implementation Ambiguity

Looking at the Keras `models_fixed.py`, the attention gates are used like:
```python
# Keras doesn't explicitly show F_g parameter
# It's implicit in the implementation
attention_block(x=skip, gating=decoder_signal, inter_shape=...)
```

In PyTorch, we need to **explicitly** specify input channel counts for Conv2D layers, which revealed this mismatch.

---

## Testing

### Verification Steps

1. **Channel size trace:**
```python
# For n_filters=32:
Bottleneck: 512 channels (32 * 16)
up4 output: 256 channels (32 * 8)  ← This goes to att4 as gating
e4:         256 channels (32 * 8)  ← This goes to att4 as skip

AttentionGate.__init__:
  F_g=256, F_l=256, F_int=128  ✓ CORRECT NOW
```

2. **Expected behavior:**
- UNet: Should complete (no attention gates) ✓
- AttentionUNet: Should now train all 27 configs
- AttentionResUNet: Should now train all 27 configs

---

## Resubmission

All three jobs need to be resubmitted:

```bash
# Clean up partial results (optional)
# rm -rf pytorch_comparison_no_aug_20251020_212738
# rm -rf pytorch_comparison_with_aug_20251020_213146
# rm -rf pytorch_comparison_adaptive_loss_20251020_213148

# Resubmit with fixed code
qsub pbs_train_pytorch_comparison_no_aug.sh
qsub pbs_train_pytorch_comparison_with_aug.sh
qsub pbs_train_pytorch_comparison_adaptive_loss.sh
```

**Expected runtime:** ~18-24 hours per job (all 81 models should complete now)

---

## Lessons Learned

### 1. Channel Tracking is Critical
When porting models between frameworks, **trace tensor shapes** at every step, especially through:
- Upsampling operations (reduce channels)
- Downsampling operations (increase channels)
- Skip connections (must match dimensions)

### 2. Test Small First
The error could have been caught earlier by:
```python
# Quick test with single config
CONFIG['architectures'] = ['unet', 'attention_unet']  # Test both
CONFIG['hyperparam_grid'] = {
    'n_filters': [32],  # Single value
    'dropout': [0.2],
    'learning_rate': [0.001],
}
# Run locally with epochs=2
```

### 3. Framework Differences
- **Keras:** Implicit channel handling in layers
- **PyTorch:** Explicit channel counts in `__init__`
- PyTorch's explicitness catches this type of error earlier, but requires more care

---

## Summary

**Problem:** Attention gate expected wrong number of input channels (bottleneck size instead of upsampled size)

**Root Cause:** `F_g` parameter used pre-upsampling channels instead of post-upsampling channels

**Fix:** Changed `F_g` to match upsampled decoder channels at each level

**Impact:** All 6 attention-based model classes across 3 files fixed

**Status:** ✅ Ready for resubmission

---

**Bug Report Date:** October 21, 2025
**Fix Applied:** October 21, 2025
**Files Fixed:** 3 (all PyTorch comparison training scripts)
**Architectures Fixed:** AttentionUNet, AttentionResUNet
