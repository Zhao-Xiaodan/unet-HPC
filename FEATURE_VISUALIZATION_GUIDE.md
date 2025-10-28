# Feature Visualization for U-Net - Complete Guide

## Overview

This implementation brings **optimization-based feature visualization** from the Distill 2017 article "Feature Visualization" (Olah et al.) to your U-Net microscopy segmentation model.

**Paper**: https://distill.pub/2017/feature-visualization/

## Core Concept: Understanding Neural Networks Through Visualization

### The Main Idea

Instead of looking at what real images activate neurons, we **create synthetic images** that maximally activate specific features. This reveals what the network has learned to detect.

**Analogy**: It's like asking "What would be your dream input?" to each neuron.

`★ Insight ─────────────────────────────────────────────────────────────`
**Why this matters for your U-Net:**
1. **Debugging**: See if the model learns meaningful microbead features
2. **Understanding**: Discover what patterns distinguish positive detections
3. **Validation**: Verify the model isn't relying on artifacts or noise
`───────────────────────────────────────────────────────────────────────`

## Key Techniques from the Article

### 1. Optimization-Based Visualization

**Core method**: Gradient ascent in pixel space

```python
# Start with random noise
image = random_noise()

# Iteratively modify image to maximize activation
for iteration in range(500):
    activation = model(image)[layer][channel]
    loss = -activation.mean()  # Negative for maximization
    loss.backward()
    image = image + learning_rate * image.grad
```

**What you get**: An image that makes the target neuron/channel "fire" as strongly as possible.

### 2. Regularization Techniques

Raw optimization often produces noisy, uninterpretable patterns. The article presents several regularization methods:

#### **A. L2 Penalty** (Prevent extreme values)
```python
l2_loss = lambda_l2 * (image ** 2).mean()
```
- Keeps pixel values reasonable
- Prevents saturation

#### **B. Total Variation** (Encourage smoothness)
```python
tv_loss = |∇x image| + |∇y image|
```
- Penalizes rapid changes between adjacent pixels
- Creates spatially coherent patterns

#### **C. Gaussian Blur** (Reduce high-frequency noise)
```python
if iteration % 4 == 0:
    image = gaussian_blur(image, sigma=0.5)
```
- Applied periodically during optimization
- Removes "salt-and-pepper" noise

#### **D. Jitter** (Translation robustness)
```python
# Randomly shift image slightly each iteration
offset_x, offset_y = random.randint(-8, 8, size=2)
image_shifted = shift(image, offset_x, offset_y)
```
- Encourages translation-invariant features
- Prevents edge artifacts

#### **E. Random Transformations**
- Small rotations (±5°)
- Slight scaling
- Helps find more robust features

### 3. Diversity Techniques

**Problem**: One channel can respond to multiple different patterns.

**Solution**: Generate diverse examples from different random initializations

```python
for seed in [0, 1, 2]:  # 3 diverse examples
    torch.manual_seed(seed)
    image_i = visualize_channel(layer, channel, seed=seed)
```

**Result**: Reveals the full range of what a channel detects.

### 4. DeepDream

**Variant**: Start from a real image and amplify what the network sees

```python
# Instead of random noise, use real image
image = real_microscopy_image.copy()

# Maximize activation (same as before)
for iteration in range(100):
    activation = model(image)[layer]
    loss = -activation.mean()  # Amplify ALL features
    loss.backward()
    image = image + lr * image.grad
```

**Multi-octave DeepDream**:
- Process at multiple scales (coarse → fine)
- Amplifies patterns at different spatial frequencies
- Creates recursive, fractal-like enhancements

## What Different Layers Learn

Based on the Distill article's findings for image models:

### **Early Layers** (encoder_1, encoder_2)
- **Edges and lines** at various orientations
- **Simple textures**: dots, grids, waves
- **Color/brightness** gradients
- **Local features**: small-scale patterns

### **Middle Layers** (encoder_3, encoder_4)
- **Textures**: repeating patterns
- **Object parts**: corners, junctions, specific shapes
- **Combinations** of low-level features
- **Context-aware** patterns

### **Deep Layers** (bottleneck)
- **Semantic concepts**: object-level features
- **Complex patterns**: combinations of mid-level features
- **Task-specific**: features relevant to segmentation
- **Abstract representations**: hard to describe verbally

### **Decoder Layers** (decoder_1, decoder_2, etc.)
- **Reconstruction patterns**: how to rebuild spatial structure
- **Boundary features**: edge refinement
- **Multi-scale integration**: combining encoder and decoder info
- **Output-specific**: patterns that produce good segmentations

## Implementation for Your U-Net

### Architecture Overview

```
Input [1×512×512]
    ↓
Encoder_1 [32 channels]  ← Detects edges, simple patterns
    ↓ pool
Encoder_2 [64 channels]  ← Combines edges into textures
    ↓ pool
Encoder_3 [128 channels] ← Object parts, complex textures
    ↓ pool
Encoder_4 [256 channels] ← High-level features
    ↓ pool
Bottleneck [512 channels] ← Most abstract representation
    ↓ upsample
Decoder_4 [256 channels] ← Begin reconstruction
    ↓ upsample + skip
Decoder_3 [128 channels] ← Refine boundaries
    ↓ upsample + skip
Decoder_2 [64 channels]  ← Spatial refinement
    ↓ upsample + skip
Decoder_1 [32 channels]  ← Final detail recovery
    ↓
Output [1×512×512]       ← Segmentation mask
```

### What to Visualize

**Recommended layers**:
1. `encoder_1_conv2` - What are the basic building blocks?
2. `encoder_3_conv2` - What textures does it recognize?
3. `bottleneck_conv2` - What high-level concepts exist?
4. `decoder_1_conv2` - What reconstruction patterns guide output?

**Recommended channels per layer**: 8-12 (representative sample)
**Diverse examples per channel**: 3 (shows range of patterns)

### Output Structure

```
unet_feature_viz_YYYYMMDD_HHMMSS/
├── metadata.json                                    # Experiment info
├── encoder_1_conv2_diverse_visualizations.png      # Grid view
├── encoder_1_conv2/
│   ├── ch000_div0.png                              # Channel 0, example 0
│   ├── ch000_div0_history.png                      # Optimization curve
│   ├── ch000_div1.png                              # Channel 0, example 1
│   ├── ch000_div2.png                              # Channel 0, example 2
│   ├── ch001_div0.png                              # Channel 1, example 0
│   └── ...
├── encoder_3_conv2_diverse_visualizations.png
├── encoder_3_conv2/
│   └── ...
├── bottleneck_conv2_diverse_visualizations.png
├── bottleneck_conv2/
│   └── ...
├── decoder_1_conv2_diverse_visualizations.png
├── decoder_1_conv2/
│   └── ...
└── deepdream/
    ├── image1_encoder_1_conv2_deepdream.png
    ├── image1_bottleneck_conv2_deepdream.png
    └── ...
```

## Usage Instructions

### Basic Usage

```bash
cd /scratch/phyzxi/unet-HPC

# Submit job to HPC
qsub pbs_feature_visualization.sh

# Check status
qstat -u phyzxi

# Monitor progress
tail -f UNet_Feature_Viz.o<jobid>
```

### Advanced Usage (Custom Parameters)

```bash
# Visualize specific layers only
python unet_feature_visualization.py \
    --model_path ./best_models_PyTorch/unet/best_model.pth \
    --layers encoder_1_conv2 bottleneck_conv2 \
    --channels_per_layer 16 \
    --diverse_per_channel 5 \
    --iterations 1000

# Quick exploration (fewer iterations)
python unet_feature_visualization.py \
    --channels_per_layer 4 \
    --diverse_per_channel 2 \
    --iterations 200

# High-quality visualizations
python unet_feature_visualization.py \
    --iterations 1000 \
    --image_size 1024 \
    --channels_per_layer 16

# DeepDream only
python unet_feature_visualization.py \
    --layers decoder_1_conv2 \
    --channels_per_layer 0 \
    --deepdream \
    --test_images_dir ./test_images
```

### Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | `./best_models_PyTorch/unet/best_model.pth` | Trained model checkpoint |
| `--output_dir` | `./unet_feature_viz` | Output directory |
| `--n_filters` | 32 | U-Net base filters |
| `--dropout` | 0.2 | Dropout rate (must match training) |
| `--layers` | [list] | Layers to visualize |
| `--channels_per_layer` | 8 | Number of channels per layer |
| `--diverse_per_channel` | 3 | Diverse examples per channel |
| `--iterations` | 500 | Optimization iterations |
| `--image_size` | 512 | Generated image size |
| `--deepdream` | False | Enable DeepDream |
| `--test_images_dir` | `./test_images` | Images for DeepDream |

## Interpreting Results

### Grid Visualizations

Each grid shows:
- **Rows**: Different channels in the layer
- **Columns**: Diverse examples for that channel
- **Title**: Channel index + activation value

**What to look for**:
- ✅ Clear, interpretable patterns → Model learns meaningful features
- ❌ Random noise → Possible overfitting or poor convergence
- ✅ Diverse examples look related → Channel has consistent role
- ❌ Diverse examples unrelated → Channel may be ambiguous

### Individual Channel Images

**Early layers** (encoder_1, encoder_2):
- Expect: Edge detectors, simple oriented patterns
- Example: Horizontal lines, vertical edges, diagonal gradients
- Microbead context: Circle arcs, bright-to-dark transitions

**Middle layers** (encoder_3, encoder_4):
- Expect: Texture patterns, repeated motifs
- Example: Grid patterns, dot arrays, radiating lines
- Microbead context: Circular patterns, particle boundaries, spacing patterns

**Bottleneck**:
- Expect: Complex, abstract features (may be hard to interpret)
- Example: Multi-scale patterns, combinations of lower-level features
- Microbead context: Full particle representations, density patterns

**Decoder layers**:
- Expect: Reconstruction templates, boundary refiners
- Example: Circular outlines, edge sharpeners, spatial smoothers
- Microbead context: How to draw segmentation boundaries

### Activation Histories

**Plot shows**: Activation value vs. optimization iteration

**Good optimization**:
- ✅ Steady increase, plateaus at high value
- ✅ Smooth curve (regularization working)

**Problem signs**:
- ❌ Flat or decreasing → May need higher learning rate
- ❌ Noisy, erratic → Reduce learning rate or increase regularization
- ❌ Very low final value → Channel may be dead/unused

### DeepDream Results

**What it shows**: Patterns the network "sees" in real images

**Interpretation**:
- **Enhanced edges** → Network focuses on boundaries
- **Amplified textures** → Network detects specific patterns
- **Artifact creation** → Network invents patterns (caution: may indicate bias)

**Useful for**:
- Understanding what the model prioritizes
- Detecting spurious correlations
- Validating that model looks at meaningful regions

## Expected Runtime

### Per Channel Visualization

- **Iterations**: 500
- **Image size**: 512×512
- **Time**: ~10-15 seconds on GPU

### Full Job Estimate

**Default configuration** (6 layers × 12 channels × 3 diverse):
- Total visualizations: 216
- Time per viz: ~12 seconds
- **Total**: ~45 minutes

**With DeepDream** (3 test images × 6 layers):
- Additional: ~18 visualizations
- Time per dream: ~30 seconds
- **Additional**: ~10 minutes

**Total expected runtime**: **~1 hour**

## Scientific Insights

### What You Can Learn

1. **Feature Hierarchy**
   - Do early layers learn edge detectors?
   - Do deep layers learn semantic concepts?
   - How does information flow through the network?

2. **Microbead-Specific Patterns**
   - What distinguishes a microbead from background?
   - What patterns indicate particle boundaries?
   - How does the model handle overlapping particles?

3. **Skip Connection Role**
   - Compare encoder vs. decoder features at same resolution
   - How do skip connections preserve spatial information?
   - What's lost and recovered during downsampling/upsampling?

4. **Model Validation**
   - Are learned features interpretable?
   - Does the model rely on artifacts?
   - Are there unexpected biases?

### Potential Discoveries

**Good signs**:
- ✅ Encoder_1 detects edges and gradients
- ✅ Middle layers combine edges into circular patterns
- ✅ Bottleneck represents full particles
- ✅ Decoder refines boundaries and fills regions

**Warning signs**:
- ⚠️ Many channels produce random noise
- ⚠️ Features unrelated to task (detecting image borders, etc.)
- ⚠️ Over-reliance on specific artifacts
- ⚠️ Dead channels (very low activation)

## Comparison with CRP

### Feature Visualization vs CRP

| Aspect | Feature Visualization | CRP (Your Previous Work) |
|--------|----------------------|--------------------------|
| **Question** | What does this channel detect? | How do channels influence each other? |
| **Method** | Optimize synthetic image | Trace gradients through network |
| **Output** | Interpretable image | Dependency graph |
| **Reveals** | Learned features | Information flow |
| **Use case** | Understanding filters | Debugging decisions |

### Complementary Insights

**Use both together**:
1. **Feature viz** → "Channel 42 detects circular boundaries"
2. **CRP** → "Channel 42 strongly influences decoder_1 Ch19"
3. **Conclusion** → "Boundary detection in encoder propagates to output refinement"

## Troubleshooting

### Issue: Visualizations are noisy/uninterpretable

**Solutions**:
1. Increase total variation weight: `regularization_config['tv_weight'] = 0.1`
2. Increase blur frequency: `regularization_config['blur_every'] = 2`
3. More iterations: `--iterations 1000`
4. Lower learning rate: `lr=0.01` instead of `0.05`

### Issue: Optimization doesn't converge

**Check**:
1. Activation history plot - is it increasing?
2. Layer name correct? Check available layers
3. Channel index valid? (0 to num_channels-1)

**Solutions**:
- Increase learning rate: `lr=0.1`
- More iterations: `--iterations 1000`
- Try different channel (some may be dead)

### Issue: All visualizations look similar

**Possible causes**:
1. Not enough diversity: Increase `--diverse_per_channel 5`
2. Over-regularization: Reduce TV weight
3. Channels genuinely similar (early layers often are)

### Issue: DeepDream produces artifacts

**This is actually informative!**
- Shows what network hallucinates
- May indicate training biases
- Can reveal spurious correlations

**To reduce artifacts**:
- Fewer iterations: `iterations=50`
- Lower learning rate: `lr=0.005`
- Fewer octaves: `num_octaves=2`

## Next Steps

After running feature visualization:

1. **Analyze layer-by-layer progression**
   - How do features evolve from input to output?
   - Where does semantic understanding emerge?

2. **Compare with CRP results**
   - Do highly-connected channels (from CRP) show related features?
   - Can you trace a feature path through the network?

3. **Validate model behavior**
   - Are learned features sensible for microbead segmentation?
   - Any unexpected patterns?

4. **Generate publication figures**
   - Grid visualizations show what the model learned
   - DeepDream shows what model "sees" in real images
   - Combine with CRP for complete story

5. **Guide model improvements**
   - Dead channels → Reduce capacity
   - Noisy features → Increase regularization during training
   - Missing expected features → Augment training data

## Additional Resources

### Related Techniques

**From the Distill article series**:
- **Activation Atlases**: Clustering similar activations
- **Neural Style Transfer**: Using learned features for art
- **Interpretable Neurons**: Finding "grandmother cells"

**Other visualization methods**:
- **Grad-CAM**: Highlight important regions in input
- **Saliency maps**: Where does model look?
- **Adversarial examples**: Find model vulnerabilities

### Further Reading

- Original article: https://distill.pub/2017/feature-visualization/
- Distill threads: https://distill.pub/2018/building-blocks/
- OpenAI Microscope: https://microscope.openai.com/

---

**Status**: ✅ Ready to run
**Files created**:
- `unet_feature_visualization.py` - Main implementation
- `pbs_feature_visualization.sh` - HPC submission script
- `FEATURE_VISUALIZATION_GUIDE.md` - This guide

**To start**: `qsub pbs_feature_visualization.sh`

**Questions to answer**:
- What patterns activate microbead detector channels?
- How does the U-Net represent particles internally?
- What reconstruction strategies does the decoder use?

Discover what your U-Net has learned! 🔬
