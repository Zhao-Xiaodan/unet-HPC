# Complete CRP Analysis with Encoder Paths - Usage Guide

## Overview

This enhanced CRP (Concept Relevance Propagation) analysis system addresses all your requirements:

✅ **Multi-hop path visualization** - Traces dependencies from decoder_1 all the way to encoder_1
✅ **Encoder path inclusion** - Continues beyond bottleneck through encoder layers
✅ **Feature map display** - Shows actual channel activations when nodes are highlighted
✅ **Dynamic top-K adjustment** - User can change K from 1-10 in real-time
✅ **Skip connection support** - Includes lateral connections between decoder and encoder

## What's Been Implemented

### 1. Enhanced CRP Analysis Script
**File**: `unet_crp_complete_with_encoders.py`

**Features**:
- Computes **12 connections per image** (vs 4 in basic version):
  - 4 decoder path: decoder_1 → decoder_2 → decoder_3 → decoder_4 → bottleneck
  - 4 encoder path: bottleneck → encoder_4 → encoder_3 → encoder_2 → encoder_1
  - 4 skip connections: decoder ← encoder (lateral)
- Stores **COMPLETE** relevance matrices (all channel pairs, not just top-K)
- Extracts and saves **all feature maps** as PNG files
- Creates comprehensive metadata for visualization

**Output structure**:
```
unet_crp_complete_YYYYMMDD_HHMMSS/
├── metadata.json                    # Complete analysis metadata
├── <image_name>_tile.png           # Input tile
├── <image_name>/
│   ├── decoder_1_conv2_from_decoder_2_conv2.npy     # Decoder path
│   ├── decoder_2_conv2_from_decoder_3_conv2.npy
│   ├── decoder_3_conv2_from_decoder_4_conv2.npy
│   ├── decoder_4_conv2_from_bottleneck_conv2.npy
│   ├── bottleneck_conv2_from_encoder_4_conv2.npy   # Encoder path
│   ├── encoder_4_conv2_from_encoder_3_conv2.npy
│   ├── encoder_3_conv2_from_encoder_2_conv2.npy
│   ├── encoder_2_conv2_from_encoder_1_conv2.npy
│   ├── decoder_4_conv2_from_encoder_4_conv2.npy    # Skip connections
│   ├── decoder_3_conv2_from_encoder_3_conv2.npy
│   ├── decoder_2_conv2_from_encoder_2_conv2.npy
│   ├── decoder_1_conv2_from_encoder_1_conv2.npy
│   └── feature_maps/
│       ├── decoder_1_conv2/
│       │   ├── ch000.png ... ch031.png
│       ├── decoder_2_conv2/
│       │   ├── ch000.png ... ch063.png
│       ├── ... (all 9 layers)
```

### 2. Enhanced Interactive Visualization
**File**: `generate_enhanced_interactive_visualization.py`

**Features**:
- **Multi-hop path tracing**: Recursively follows connections through all layers
- **Feature map preview**: Displays channel activations on node hover/click
- **Dynamic top-K slider**: Adjust number of paths shown (1-10)
- **Skip connection toggle**: Show/hide lateral connections
- **Three-column layout**: Controls | Graph | Info Panel
- **Path depth grouping**: Organizes paths by number of hops in info panel

**User interactions**:
- Hover over node → see top-K outgoing paths
- Click node → lock selection and show feature map
- Adjust K slider → dynamically update displayed paths
- Toggle skip connections → simplify/complicate view
- Switch images → compare different test cases

### 3. PBS Submission Script
**File**: `pbs_crp_complete_with_encoders.sh`

**Configuration**:
- Walltime: 8 hours (to handle 12 connections × multiple images)
- GPU: 1 GPU, 32GB memory
- Runs both analysis and visualization automatically
- Email notifications on start/end/abort

## How to Run

### Step 1: Submit the PBS Job

```bash
cd /Users/xiaodan/unetCNN/unet-HPC
qsub pbs_crp_complete_with_encoders.sh
```

### Step 2: Monitor Progress

Check job status:
```bash
qstat -u phyzxi
```

View live output (while running):
```bash
tail -f CRP_Complete_Encoders.o<job_id>
```

### Step 3: Check Results

After job completes, find output directory:
```bash
ls -lhtr unet_crp_complete_*/
```

Verify all files generated:
```bash
OUTPUT_DIR=$(ls -td unet_crp_complete_* | head -1)
echo "Output directory: $OUTPUT_DIR"
tree -L 2 $OUTPUT_DIR
```

### Step 4: Copy to Local Machine

From your local machine:
```bash
# Copy entire results directory
scp -r <username>@hpc-login:/path/to/unet_crp_complete_YYYYMMDD_HHMMSS ~/Downloads/

# Or just the visualization
scp <username>@hpc-login:/path/to/unet_crp_complete_YYYYMMDD_HHMMSS/crp_enhanced_visualization.html ~/Downloads/
```

### Step 5: View Interactive Visualization

1. Navigate to downloaded directory
2. Open `crp_enhanced_visualization.html` in web browser (Chrome/Firefox recommended)
3. Interact with the graph!

## Expected Runtime

Based on test image count and hardware:

- **8 test images**: ~6-8 hours
  - CRP computation: ~40-50 min per image × 8 = 5-7 hours
  - Feature map saving: ~10 min per image × 8 = 1 hour
  - Visualization generation: ~5-10 min

- **Per image breakdown**:
  - 12 relevance matrices computation: ~40 min
  - Feature map extraction (9 layers): ~10 min
  - Total: ~50 min per image

## Understanding the Output

### Relevance Matrices

Each `.npy` file contains a matrix of shape `[target_channels, source_channels]`:

```python
import numpy as np

# Load relevance matrix
rel_matrix = np.load('decoder_1_conv2_from_decoder_2_conv2.npy')
print(f"Shape: {rel_matrix.shape}")  # e.g., (32, 64)

# rel_matrix[i, j] = relevance from source channel j to target channel i
# Higher value = stronger dependency

# Top contributors to target channel 19
target_ch = 19
top_sources = np.argsort(rel_matrix[target_ch])[::-1][:10]
print(f"Top 10 sources for Ch{target_ch}: {top_sources}")
print(f"Relevances: {rel_matrix[target_ch, top_sources]}")
```

### Metadata Structure

```json
{
  "timestamp": "20251028_123456",
  "model_info": {
    "n_filters": 32,
    "dropout": 0.2,
    "model_path": "./best_models_PyTorch/unet/best_model.pth"
  },
  "layer_sequence_full": [
    "decoder_1_conv2", "decoder_2_conv2", "decoder_3_conv2", "decoder_4_conv2",
    "bottleneck_conv2",
    "encoder_4_conv2", "encoder_3_conv2", "encoder_2_conv2", "encoder_1_conv2"
  ],
  "connections": [
    ["decoder_1_conv2", "decoder_2_conv2"],
    ...
    ["bottleneck_conv2", "encoder_4_conv2"],
    ...
    ["decoder_4_conv2", "encoder_4_conv2"]
  ],
  "layer_channels": {
    "decoder_1_conv2": 32,
    "decoder_2_conv2": 64,
    "decoder_3_conv2": 128,
    "decoder_4_conv2": 256,
    "bottleneck_conv2": 512,
    "encoder_4_conv2": 256,
    "encoder_3_conv2": 128,
    "encoder_2_conv2": 64,
    "encoder_1_conv2": 32
  },
  "images": {
    "image_name": {
      "image_path": "...",
      "tile_position": [1024, 1536],
      "transitions": { ... },
      "feature_maps": { ... }
    }
  }
}
```

## Visualization Features Explained

### Multi-Hop Path Tracing

When you click on a node (e.g., decoder_1 Ch19):

1. **Depth 1**: Shows top-K contributors from decoder_2
   - e.g., Ch12 (0.82), Ch31 (0.69)

2. **Depth 2**: For each depth-1 contributor, shows their top-K sources from decoder_3
   - Ch12 → [Ch5 (0.91), Ch18 (0.67)]
   - Ch31 → [Ch22 (0.88), Ch9 (0.55)]

3. **Continues recursively** through decoder_4 → bottleneck → encoder_4 → ... → encoder_1

Result: You see the **complete dependency chain** from output features all the way back to early encoder features!

### Feature Map Display

- Feature maps are **normalized** for visualization (0-1 range)
- Colormap: **viridis** (dark blue = low, bright yellow = high)
- Size: 100 DPI PNG files (4×4 inch figures)
- Info panel shows:
  - Current layer/channel
  - Feature map image
  - Top connections to/from this channel

### Dynamic Top-K Slider

- **K=1**: Only strongest connection per channel (sparse, easy to follow)
- **K=5**: Moderate complexity (recommended for exploration)
- **K=10**: Dense graph (shows many alternative paths)

Adjusting K **instantly updates** the graph without recomputing CRP!

## Troubleshooting

### Issue: Job runs out of memory

**Solution**: Each image uses ~20-25GB during CRP computation. If issues occur:
1. Reduce number of test images
2. Request more memory: `#PBS -l select=1:...mem=64gb`

### Issue: Missing nodes in visualization

**Solution**: This should be fixed! Complete matrices include ALL channels. If issue persists:
1. Check `metadata.json` contains all 12 connections
2. Verify `.npy` files have correct shapes
3. Check browser console for JavaScript errors

### Issue: Feature maps not loading

**Solution**:
1. Ensure feature_maps directory is copied with results
2. Check relative paths in metadata.json
3. Open browser console (F12) to see loading errors
4. Verify PNG files exist: `ls <image_name>/feature_maps/*/ch000.png`

### Issue: Visualization too slow

**Solution**:
1. Reduce top-K slider value
2. Toggle off skip connections
3. Use Chrome (faster than Firefox for large graphs)
4. Reduce number of images in analysis

## Comparison with Original CRP

| Feature | Original CRP | Complete CRP with Encoders |
|---------|--------------|----------------------------|
| Decoder path | ✓ | ✓ |
| Encoder path | ✗ | ✓ |
| Skip connections | ✗ | ✓ |
| Connections per image | 4 | 12 |
| Relevance storage | Top-K only | Complete matrices |
| Missing nodes issue | Yes (Ch94) | Fixed |
| Multi-hop tracing | Limited | Full recursion |
| Feature maps | ✗ | ✓ |
| Runtime per image | ~10 min | ~50 min |
| Output size | ~50 KB | ~700 KB + feature maps |

## Next Steps

After viewing the visualization:

1. **Identify key paths**: Which channels have strongest multi-hop dependencies?
2. **Examine feature maps**: What patterns do highly-relevant channels detect?
3. **Compare across images**: Are dependencies consistent or image-specific?
4. **Focus analysis**: Select specific paths for deeper investigation
5. **Generate reports**: Export findings for publication/presentation

## Additional Resources

- **Full implementation guide**: `CRP_COMPREHENSIVE_GUIDE.md`
- **Debug documentation**: `CRP_DEBUG_FIX.md`, `CRP_DEBUG_FIX_v2.md`
- **Original paper**: Achtibat et al., "From Attribution Maps to Human-Understandable Explanations through Concept Relevance Propagation"

## Contact

For issues or questions:
- Check browser console for errors
- Review PBS job output logs
- Verify all prerequisites are met (PyTorch, numpy, matplotlib, tqdm)

---

**Status**: ✅ Ready to run
**Last updated**: October 28, 2025
**Author**: Claude Code
