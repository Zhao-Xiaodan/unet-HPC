# U-Net Concept Relevance Propagation (CRP) Analysis

## Overview

This implementation adapts **Concept Relevance Propagation (CRP)** from the Nature Machine Intelligence 2023 paper to analyze hierarchical concept composition in U-Net architecture for microbead segmentation.

### What is CRP?

CRP extends Layer-wise Relevance Propagation (LRP) to understand:
1. **What concepts (feature channels) contribute to higher-level concepts**
2. **How concepts compose hierarchically through network layers**
3. **Which spatial regions activate specific concepts**

### Our Analysis Target

Based on PCA cluster analysis in `unet_visualization_advanced_20251028_091857`:
- **Start:** `decoder_1_conv2` Ch4 (Cluster 6) - identified as edge detection channel
- **Goal:** Trace which channels in decoder_2, decoder_3, decoder_4, and bottleneck contribute to this edge detector

## Files Created

### 1. `unet_crp_hierarchical_concepts.py`

Main Python script implementing CRP for U-Net:

**Key Features:**
- **UNetCRP class:** Implements conditional relevance propagation adapted for U-Net
- **Handles skip connections:** Properly traces relevance through concatenation operations
- **Hierarchical tracing:** Automatically traces from target channel back through decoder layers
- **Visualization:** Generates concept composition graphs showing channel dependencies

**Key Methods:**
- `conditional_relevance_propagation()`: Computes relevance scores using gradient-based attribution
- `trace_hierarchical_concepts()`: Traces concept composition through multiple layers
- `visualize_hierarchy()`: Creates hierarchical graph visualization

### 2. `pbs_unet_crp_analysis.sh`

PBS script for HPC execution:
- Walltime: 2 hours
- Resources: 1 GPU, 8 CPUs, 32GB RAM
- Automatically finds latest model and generates analysis

## How to Run

### Method 1: HPC Cluster (Recommended)

```bash
# Submit to PBS queue
qsub pbs_unet_crp_analysis.sh

# Monitor job
qstat -u $USER

# Check output
tail -f UNet_CRP_Analysis.o<jobid>
```

### Method 2: Local Execution (CPU/GPU)

```bash
# Make sure you have a trained model
python unet_crp_hierarchical_concepts.py \
    --model_path ./best_models_PyTorch/unet/best_model.pth \
    --test_image ./test_images/320x_2025-05-15_02-05-00.tif \
    --start_layer decoder_1_conv2 \
    --start_channel 4 \
    --top_k 2 \
    --n_filters 32 \
    --dropout 0.2
```

### Method 3: Analyze Different Channels

To analyze other channels from the PCA clusters:

```bash
# Example: Analyze Ch16 (also in Cluster 6)
python unet_crp_hierarchical_concepts.py \
    --start_channel 16 \
    --start_layer decoder_1_conv2

# Example: Analyze Ch10 from encoder_1
python unet_crp_hierarchical_concepts.py \
    --start_channel 10 \
    --start_layer encoder_1_conv2
```

## Output Files

The analysis creates a timestamped directory `unet_crp_analysis_YYYYMMDD_HHMMSS/` containing:

### 1. `input_tile.png`
- The 512×512 input tile used for analysis
- Extracted from row 3, column 4 of test image

### 2. `hierarchy.json`
JSON file with complete hierarchical concept data:
```json
{
  "decoder_1_conv2": {
    "channels": [4],
    "contributes_from": {
      "layer": "decoder_2_conv2",
      "channels": [12, 31],
      "relevance": [0.8234, 0.6891]
    }
  },
  ...
}
```

### 3. `hierarchical_concept_graph.png`
Visual graph showing:
- **Nodes:** Channel numbers at each layer
- **Edges:** Relevance flow (thickness = relevance strength)
- **Labels:** Relevance scores on connections
- **Layout:** Left (shallow/decoder_1) → Right (deep/bottleneck)

## Expected Results

### Example Hierarchical Tracing for Ch4 (Edge Detector)

```
decoder_1_conv2: Ch4 (Edge Detection - Yellow Edges)
    ↓ (relevance: 0.82, 0.69)
decoder_2_conv2: Ch12, Ch31
    ↓ (relevance: 0.75, 0.58)
decoder_3_conv2: Ch45, Ch23
    ↓ (relevance: 0.68, 0.52)
decoder_4_conv2: Ch89, Ch67
    ↓ (relevance: 0.61, 0.48)
bottleneck_conv2: Ch234, Ch156
```

This shows:
1. **Ch4's edge detection** in decoder_1 is primarily composed of **Ch12 and Ch31** from decoder_2
2. These in turn depend on **Ch45 and Ch23** from decoder_3
3. And so on, tracing back to the bottleneck

## Technical Details

### CRP Implementation Strategy

Since full zennit-based LRP is complex for U-Net with skip connections, we use:

**Gradient-Based Approximation:**
1. **Forward pass:** Compute activations at all layers
2. **Conditional signal:** Select only target channels
3. **Backward pass:** Compute gradients w.r.t. source layer
4. **Relevance:** activation × gradient (Grad-CAM style)

This is mathematically related to CRP's conditional relevance:
```
R^(l-1)_i ← (activation_i × gradient_i) | condition on channel_j^(l)
```

### Handling Skip Connections

U-Net decoder concatenates features from:
1. **Upsampling path:** Features from deeper layer
2. **Skip connection:** Features from encoder at same resolution

Our implementation:
- Traces backward through ConvBlock layers (conv2 → conv1)
- Handles concatenation by computing gradients through both paths
- Aggregates relevance from multiple source channels

### Why Gradient-Based?

**Advantages:**
- ✓ Fast computation (single backward pass)
- ✓ Works with any PyTorch model
- ✓ Handles complex operations (BatchNorm, concatenation, etc.)
- ✓ Stable numerical properties

**Limitations:**
- ✗ Not exact LRP (approximation)
- ✗ May not satisfy conservation property
- ✗ Sensitive to gradient saturation

For more accurate results, future work could implement:
- Full LRP rules with custom propagation for concatenation
- Deep Taylor Decomposition for nonlinear operations
- Layer-wise ε-LRP with skip connection handling

## Interpretation Guide

### Reading the Concept Graph

**Node Size/Color:**
- Represents the channel at that layer
- Blue = active channels in hierarchy

**Edge Thickness:**
- Thicker = stronger relevance contribution
- Represents how much target channel depends on source channel

**Relevance Values:**
- Range: 0.0 to 1.0 (normalized)
- > 0.7: Strong dependency
- 0.4-0.7: Moderate dependency
- < 0.4: Weak dependency

### Understanding the Results

**High Relevance (>0.7):**
- Source channel is **critical** for target concept
- Removing it would significantly change target activation
- Indicates strong compositional relationship

**Medium Relevance (0.4-0.7):**
- Source channel **contributes** but is not essential
- Part of ensemble of features
- May be redundant with other channels

**Low Relevance (<0.4):**
- Weak contribution
- May be coincidental or indirect
- Consider ignoring for concept interpretation

## Comparison with CRP Paper (Fig. 3)

### What We Implement

Similar to CRP paper Fig. 3, we provide:

✓ **Conditional heatmaps:** Spatial relevance showing where concepts activate
✓ **Hierarchical composition:** Layer-by-layer concept tracing
✓ **Relevance quantification:** Numerical scores for channel contributions

### What We Don't Implement (Yet)

Future extensions could add:

⚬ **RelMax reference samples:** Find images that maximize relevance (not just activation)
⚬ **Masked visualizations:** Show only relevant regions for each concept
⚬ **Concept descriptions:** Automatic labeling of what each channel detects
⚬ **Multi-conditional analysis:** How multiple concepts interact

## Troubleshooting

### Model Not Found

```
ERROR: Model not found: ./best_models_PyTorch/unet/best_model.pth
```

**Solution:**
1. Train a model first using `qsub pbs_train_pytorch_comparison_no_aug.sh`
2. Or run visualization which caches best models: `qsub pbs_unet_visualization_advanced.sh`
3. Or specify correct model path with `--model_path`

### Test Image Not Found

```
ERROR: Test image not found: ./test_images/320x_2025-05-15_02-05-00.tif
```

**Solution:**
1. Check available test images: `ls test_images/`
2. Update `TEST_IMAGE` in PBS script
3. Or use `--test_image` flag

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Use smaller tile size (modify `extract_center_tile`)
2. Reduce batch size (not applicable here - batch=1)
3. Use CPU instead: set `device='cpu'` in script

### No Relevance Flow

```
WARNING: All relevance values near zero
```

**Solution:**
1. Check if model is properly loaded (not random initialization)
2. Verify target channel is actually active for this input
3. Try different input image with stronger features

## Citation

If you use this CRP implementation, please cite:

**CRP Method:**
```
@article{achtibat2023crp,
  title={From attribution maps to human-understandable explanations through Concept Relevance Propagation},
  author={Achtibat, Reduan and others},
  journal={Nature Machine Intelligence},
  year={2023}
}
```

**U-Net Architecture:**
```
@article{ronneberger2015unet,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  journal={MICCAI},
  year={2015}
}
```

## Contact

For questions or issues with this CRP implementation, please check:
1. This README file
2. Comments in `unet_crp_hierarchical_concepts.py`
3. Original CRP paper: https://www.nature.com/articles/s42256-023-00711-8
4. Zennit documentation: https://github.com/chr5tphr/zennit

## Version History

- **v1.0** (2025-10-28): Initial implementation with gradient-based CRP
  - Hierarchical concept tracing
  - Visualization of concept composition graphs
  - Support for arbitrary starting layer/channel
