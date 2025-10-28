# CRP Complete Analysis Debug Fix v3

## Error Summary

**Job**: CRP_Complete_Encoders.o327790
**Date**: October 28, 2025
**Status**: ❌ Failed during feature map extraction
**Progress**: ✅ Successfully computed all 12 relevance matrices before failure

## Error Details

### What Worked ✅

The analysis **successfully completed** all 12 relevance matrix computations:

**Decoder Path (4 connections)**:
- ✅ decoder_1_conv2 ← decoder_2_conv2 (32×64)
- ✅ decoder_2_conv2 ← decoder_3_conv2 (64×128)
- ✅ decoder_3_conv2 ← decoder_4_conv2 (128×256)
- ✅ decoder_4_conv2 ← bottleneck_conv2 (256×512)

**Encoder Path (4 connections)**:
- ✅ bottleneck_conv2 ← encoder_4_conv2 (512×256)
- ✅ encoder_4_conv2 ← encoder_3_conv2 (256×128)
- ✅ encoder_3_conv2 ← encoder_2_conv2 (128×64)
- ✅ encoder_2_conv2 ← encoder_1_conv2 (64×32)

**Skip Connections (4 lateral connections)**:
- ✅ decoder_4_conv2 ← encoder_4_conv2 (256×256)
- ✅ decoder_3_conv2 ← encoder_3_conv2 (128×128)
- ✅ decoder_2_conv2 ← encoder_2_conv2 (64×64)
- ✅ decoder_1_conv2 ← encoder_1_conv2 (32×32)

**Total progress**: 12/12 relevance matrices computed and saved 🎉

### What Failed ❌

Feature map extraction crashed with:

```
RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)
should be the same or input should be a MKLDNN tensor and weight is a dense tensor
```

**Error location**: `unet_crp_complete_with_encoders.py:249`

```python
File "/scratch/phyzxi/unet-HPC/unet_crp_complete_with_encoders.py", line 249, in save_feature_maps
    output, intermediates = self.model(input_tensor, return_intermediates=True)
```

## Root Cause Analysis

### The Device Mismatch Problem

PyTorch requires **tensors and models to be on the same device** (CPU or GPU).

**What happened**:

1. **Model initialized on GPU** ✓
   ```python
   crp = CompleteCRP(model, device='cuda')
   self.model = model.to(device)  # Model on GPU
   ```

2. **CRP computation moves input to GPU** ✓
   ```python
   def compute_full_relevance_matrix(self, input_tensor, ...):
       input_tensor = input_tensor.to(self.device).requires_grad_(True)  # Creates NEW tensor on GPU
       output, intermediates = self.model(input_tensor, ...)  # Works!
   ```

3. **Feature map extraction uses ORIGINAL tensor** ✗
   ```python
   def save_feature_maps(self, input_tensor, output_dir):
       # input_tensor is STILL on CPU (original, not the GPU copy from step 2)
       output, intermediates = self.model(input_tensor, ...)  # CRASH!
   ```

**Key insight**: `.to(device)` creates a **new tensor** - it doesn't modify the original!

### Why This Wasn't Caught Earlier

- ✅ `compute_full_relevance_matrix`: Each call moves input to GPU internally → works fine
- ✅ Called 12 times successfully for all relevance matrices
- ❌ `save_feature_maps`: Receives original CPU tensor → crashes

The bug only manifests when `save_feature_maps` is called!

## The Fix

### Code Change

**File**: `unet_crp_complete_with_encoders.py`
**Line**: 247-252

**BEFORE** (broken):
```python
def save_feature_maps(self, input_tensor, output_dir):
    print(f"\nExtracting feature maps...")

    # Forward pass to get all intermediates
    with torch.no_grad():
        output, intermediates = self.model(input_tensor, return_intermediates=True)
        #                                   ^^^^^^^^^^^^ CPU tensor → GPU model = CRASH!
```

**AFTER** (fixed):
```python
def save_feature_maps(self, input_tensor, output_dir):
    print(f"\nExtracting feature maps...")

    # Move input to correct device (model is on GPU, input might be on CPU)
    input_tensor = input_tensor.to(self.device)  # ← FIX: Move to GPU first!

    # Forward pass to get all intermediates
    with torch.no_grad():
        output, intermediates = self.model(input_tensor, return_intermediates=True)
        #                                   ^^^^^^^^^^^^ Now on GPU → works!
```

### Why This Fix Works

1. **Explicit device placement**: Ensures input matches model device
2. **No assumptions**: Works whether input is CPU or GPU
3. **Idempotent**: `.to(device)` on already-correct device is no-op
4. **Consistent pattern**: Matches what `compute_full_relevance_matrix` does

## Testing the Fix

### Expected Behavior After Fix

When you resubmit the job:

```bash
qsub pbs_crp_complete_with_encoders.sh
```

**Expected output**:
```
============================================================
Computing: decoder_1_conv2 ← decoder_2_conv2
============================================================
  [Progress bar] 100%
  ✓ Saved: decoder_1_conv2_from_decoder_2_conv2.npy

... [all 12 connections complete] ...

Extracting feature maps...
  Saving feature maps: 100%|██████████| 9/9 [XX:XX<00:00]
  ✓ Saved feature maps for decoder_1_conv2 (32 channels)
  ✓ Saved feature maps for decoder_2_conv2 (64 channels)
  ✓ Saved feature maps for decoder_3_conv2 (128 channels)
  ✓ Saved feature maps for decoder_4_conv2 (256 channels)
  ✓ Saved feature maps for bottleneck_conv2 (512 channels)
  ✓ Saved feature maps for encoder_4_conv2 (256 channels)
  ✓ Saved feature maps for encoder_3_conv2 (128 channels)
  ✓ Saved feature maps for encoder_2_conv2 (64 channels)
  ✓ Saved feature maps for encoder_1_conv2 (32 channels)

ANALYSIS COMPLETE
```

### Verification Steps

After job completes:

```bash
# 1. Check output directory structure
OUTPUT_DIR=$(ls -td unet_crp_complete_* | head -1)
tree -L 3 $OUTPUT_DIR

# Expected structure:
# unet_crp_complete_YYYYMMDD_HHMMSS/
# ├── metadata.json
# ├── {image_name}_tile.png
# └── {image_name}/
#     ├── decoder_1_conv2_from_decoder_2_conv2.npy
#     ├── ... (all 12 .npy files)
#     └── feature_maps/
#         ├── decoder_1_conv2/
#         ├── decoder_2_conv2/
#         ├── ... (9 layer directories)

# 2. Verify feature maps generated
ls $OUTPUT_DIR/*/feature_maps/*/ch000.png | wc -l
# Should output: 9 (one per layer)

# 3. Check metadata.json includes feature_maps info
cat $OUTPUT_DIR/metadata.json | grep -A 5 "feature_maps"

# 4. Verify all 12 relevance matrices exist
ls $OUTPUT_DIR/*/*.npy | wc -l
# Should output: 96 (12 matrices × 8 images)
```

## Performance Expectations

### Runtime Breakdown (per image)

| Stage | Time | Status |
|-------|------|--------|
| Decoder path (4 matrices) | ~8 min | ✅ Verified working |
| Encoder path (4 matrices) | ~12 min | ✅ Verified working |
| Skip connections (4 matrices) | ~8 min | ✅ Verified working |
| Feature map extraction (9 layers) | ~10 min | 🔧 Fixed |
| **Total per image** | **~38 min** | - |

### Total Job (8 images)

- **CRP + feature maps**: 38 min × 8 = ~5 hours
- **Visualization generation**: ~10 min
- **Total expected runtime**: ~5-6 hours
- **Walltime allocated**: 8 hours ✓ Sufficient buffer

## Output Size Estimates

Per image:
- **Relevance matrices**: 12 files × ~50 KB = ~600 KB
- **Feature maps**: ~1600 PNG files × 20 KB = ~32 MB
- **Total per image**: ~33 MB

Total (8 images):
- **All data**: 8 × 33 MB = **~260 MB**
- **Plus metadata/HTML**: +2 MB
- **Total output**: **~262 MB**

## Summary

✅ **Issue identified**: Device mismatch between CPU input tensor and GPU model
✅ **Fix applied**: Added `.to(self.device)` in `save_feature_maps` method
✅ **Change minimal**: Single line addition, no logic changes
✅ **Risk**: Very low - idempotent operation, consistent with existing code
✅ **Testing**: Can verify locally or resubmit PBS job

**Next step**: Resubmit the job with fixed code:

```bash
cd /scratch/phyzxi/unet-HPC
qsub pbs_crp_complete_with_encoders.sh
```

## Additional Notes

### Why This Pattern is Common

This device mismatch error is one of the most common PyTorch bugs because:

1. **Silent differences**: CPU and GPU tensors look identical
2. **Delayed failure**: Error only occurs when tensors interact
3. **Method boundaries**: Easy to forget device state when passing tensors between methods
4. **Copy semantics**: `.to()` creates new tensor, doesn't modify original

### Best Practices to Avoid This

1. **Early device placement**: Move data to device as soon as it enters the class
2. **Explicit checks**: Use assertions to verify device placement
3. **Device-agnostic code**: Use `tensor.device` instead of hardcoding 'cuda'
4. **Consistent patterns**: Always `.to(device)` before forward pass

### Example Device-Safe Pattern

```python
def process_tensor(self, input_tensor):
    # ALWAYS move to device at method entry
    input_tensor = input_tensor.to(self.device)

    # Now safe to use with model
    output = self.model(input_tensor)

    return output
```

---

**Status**: ✅ Fixed
**Version**: 3 (device mismatch in feature map extraction)
**Date**: October 28, 2025
**Author**: Claude Code
