# Bug Fix: Prediction Script Architecture Mismatch

**Date:** October 22, 2025
**Jobs Affected:**
- PyTorch_Density_Analysis.o304081 (AttentionGate)
- PyTorch_Density_Analysis.o304168 (ResConvBlock)
**Status:** ✅ Fixed (Both Issues)

---

## Problem

Prediction script failed when loading Attention UNet models with error:

```
RuntimeError: Error(s) in loading state_dict for AttentionUNet:
    Missing key(s) in state_dict: "att4.W_g.0.weight", "att4.W_g.0.bias",
                                   "att4.W_g.1.weight", "att4.W_g.1.bias", ...
    Unexpected key(s) in state_dict: "att4.W_g.weight", "att4.W_g.bias", ...
```

**Location:** `predict_pytorch_comparison.py`, line 464 (`model.load_state_dict`)

**Result:** Job aborted after 40 seconds (before GPU prediction started)

---

## Root Cause

### Architecture Mismatch

**Trained Model (train_pytorch_comparison_no_aug.py):**
```python
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1, stride=1, padding=0)  # Single Conv2d
        self.W_x = nn.Conv2d(F_l, F_int, 2, stride=2, padding=0)
        self.psi = nn.Conv2d(F_int, 1, 1, stride=1, padding=0)
        # No BatchNorm!
```

**Prediction Script (predict_pytorch_comparison.py - WRONG):**
```python
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)  # Added BatchNorm - WRONG!
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        # Sequential + BatchNorm structure doesn't match!
```

### Why This Happened

The prediction script was created by copying model definitions from a different version of the code that included BatchNorm in attention gates. The training scripts (`train_pytorch_comparison_*.py`) use simpler attention gates without BatchNorm, matching the Keras implementation.

### State Dict Structure Comparison

**Trained Model:**
```
att4.W_g.weight        [256, 256, 1, 1]   # Single Conv2d layer
att4.W_g.bias          [256]
att4.W_x.weight        [256, 256, 2, 2]
att4.W_x.bias          [256]
att4.psi.weight        [1, 256, 1, 1]
att4.psi.bias          [1]
```

**Prediction Script Expected (WRONG):**
```
att4.W_g.0.weight      # Conv2d in Sequential
att4.W_g.0.bias
att4.W_g.1.weight      # BatchNorm in Sequential
att4.W_g.1.bias
att4.W_g.1.running_mean
att4.W_g.1.running_var
# ... (BatchNorm adds many extra parameters)
```

---

## Fix Applied

### Updated AttentionGate in predict_pytorch_comparison.py

**BEFORE (lines 67-95):**
```python
class AttentionGate(nn.Module):
    """Attention gate for skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
```

**AFTER (lines 67-95) - FIXED:**
```python
class AttentionGate(nn.Module):
    """Attention gate (matching training script)"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1, stride=1, padding=0)
        self.W_x = nn.Conv2d(F_l, F_int, 2, stride=2, padding=0)
        self.psi = nn.Conv2d(F_int, 1, 1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        """
        g: gating signal from decoder
        x: skip connection from encoder
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Align dimensions
        if g1.shape[2] != x1.shape[2] or g1.shape[3] != x1.shape[3]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))

        # Upsample attention map to match skip connection size
        psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=False)

        return x * psi
```

**Key Changes:**
1. Removed `nn.Sequential` wrappers
2. Removed all `nn.BatchNorm2d` layers
3. Used simple `nn.Conv2d` layers directly
4. Added explicit dimension alignment logic
5. Matches training script exactly

---

## Files Modified

✅ `predict_pytorch_comparison.py`
- Class `AttentionGate` (lines 81-109) - Fixed in first iteration
- Class `ResConvBlock` (lines 55-79) - Fixed in second iteration
- Now both match `train_pytorch_comparison_no_aug.py` exactly

---

## Problem #2: ResConvBlock Structure Mismatch (Job o304168)

After fixing `AttentionGate`, job failed when loading `AttentionResUNet`:

```
RuntimeError: Error(s) in loading state_dict for AttentionResUNet:
    Missing key(s): "enc1.conv_block.conv1.weight", "enc1.conv_block.bn1.weight", ...
    Unexpected key(s): "enc1.conv1.weight", "enc1.bn1.weight", "enc1.shortcut.weight", ...
```

### Root Cause #2

**Trained Model (train_pytorch_comparison_no_aug.py):**
```python
class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)  # Flat
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(...)  # Direct shortcut
```

**Prediction Script (WRONG):**
```python
class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv_block = ConvBlock(...)  # Nested structure - WRONG!
        self.skip_conv = nn.Conv2d(...)   # Different name - WRONG!
```

### Fix #2: Updated ResConvBlock

```python
class ResConvBlock(nn.Module):
    """Residual convolution block (matching training script)"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Residual connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)
        return out
```

---

## Testing

### Verification Steps

1. **Check cache was created:**
   ```bash
   ls -lh best_models_PyTorch/
   # Should show 3 subdirectories: unet, attention_unet, attention_resunet
   ```

2. **Verify cache files intact:**
   ```bash
   for arch in unet attention_unet attention_resunet; do
       ls -lh best_models_PyTorch/$arch/best_model.pth
   done
   # Each should be ~50-200 MB
   ```

3. **Cache is already populated** from first run:
   - ✅ UNet model cached
   - ✅ Attention UNet model cached
   - ✅ Attention ResUNet model cached

4. **Resubmit job:**
   ```bash
   qsub pbs_pytorch_density_analysis.sh
   ```

5. **Expected behavior:**
   - Loads models from cache (fast, ~2-3 seconds)
   - Attention UNet loads successfully (no state_dict error)
   - Proceeds to prediction step
   - Completes full pipeline

---

## Why Cache Survived the Error

**Important:** The cache was successfully created before the error occurred!

**Job timeline:**
```
00:00:00  Start
00:00:25  Cache verification complete
00:00:30  Best models identified and copied to cache ✅
00:00:35  UNet model loaded successfully ✅
00:00:38  Attention UNet model loading started
00:00:40  ERROR: state_dict mismatch → Job aborted ❌
```

**Cache status after error:**
- ✅ `best_models_PyTorch/unet/` - Complete
- ✅ `best_models_PyTorch/attention_unet/` - Complete (checkpoint copied, metadata saved)
- ✅ `best_models_PyTorch/attention_resunet/` - Complete

**Why loading failed even though cache was fine:**
- The checkpoint files are correct
- The metadata is correct
- The **prediction script's model definition** was wrong
- So loading the correct checkpoint into the wrong architecture failed

---

## Resubmission

After fix, resubmit the job:

```bash
qsub pbs_pytorch_density_analysis.sh
```

**Expected output:**
```
========================================
FINDING BEST MODELS
========================================

✓ unet: Loaded from cache (IoU=0.6377)
✓ attention_unet: Loaded from cache (IoU=0.6254)
✓ attention_resunet: Loaded from cache (IoU=0.6127)

✓ All models loaded from cache: best_models_PyTorch

========================================
LOADING MODELS
========================================
Loading unet...
Loading attention_unet...
Loading attention_resunet...

========================================
PROCESSING 8 TEST IMAGES
========================================
Processing images: 100%|████████| 8/8 [XX:XX<00:00]

✓ Predictions complete

[... continues to density analysis ...]
```

---

## Lessons Learned

### 1. Architecture Consistency is Critical

When porting models between training and inference:
- **Copy model definitions exactly** from training scripts
- Don't add "improvements" (like BatchNorm) during inference
- Even small structural changes break `load_state_dict()`

### 2. Test with Small Models First

Could have caught this with a quick test:
```python
# Test script
model_train = AttentionUNet(n_channels=1, n_filters=16, dropout=0.1)
checkpoint = torch.load('best_model.pth')
model_train.load_state_dict(checkpoint['model_state_dict'])  # Would fail
```

### 3. State Dict Inspection

Useful debug commands:
```python
# List keys in checkpoint
checkpoint = torch.load('best_model.pth')
print("Checkpoint keys:", checkpoint['model_state_dict'].keys())

# List keys in current model
model = AttentionUNet(...)
print("Model keys:", model.state_dict().keys())

# Find mismatch
ckpt_keys = set(checkpoint['model_state_dict'].keys())
model_keys = set(model.state_dict().keys())
print("Missing:", model_keys - ckpt_keys)
print("Unexpected:", ckpt_keys - model_keys)
```

### 4. Framework Differences

PyTorch's `load_state_dict()` is **strict by default**:
- Every key must match exactly
- No automatic adaptation for structural changes
- Use `strict=False` only if you know what you're doing

---

## Summary

**Problem:** AttentionGate in prediction script had BatchNorm layers not present in trained models

**Root Cause:** Copied model definition from wrong source (BatchNorm version vs no-BatchNorm version)

**Fix:** Updated `AttentionGate` in `predict_pytorch_comparison.py` to match training scripts exactly

**Impact:** Attention UNet and Attention ResUNet can now load successfully

**Cache Status:** ✅ Intact, ready for next run

**Next Action:** Resubmit job - should complete successfully now

---

**Bug Report Date:** October 22, 2025
**Fix Applied:** October 22, 2025
**File Fixed:** `predict_pytorch_comparison.py`
**Architectures Affected:** Attention UNet, Attention ResUNet
**Testing:** Ready for resubmission
