# Feature Visualization - Quick Start Guide

## Summary of Distill Article Concepts

**"Feature Visualization" by Olah et al., Distill 2017**

### Core Concept

**Question**: What does a neural network "see"? What patterns activate specific neurons?

**Answer**: Generate synthetic images through optimization that maximally activate target features.

### Key Insights

1. **Optimization-based visualization**: Start with random noise, use gradient ascent to maximize activation
2. **Regularization is essential**: Raw optimization produces noise; need smoothness constraints
3. **Diversity matters**: One channel can respond to multiple patterns; generate diverse examples
4. **Layer hierarchy**: Early layers detect edges, deep layers detect concepts

### Main Techniques Implemented

| Technique | Purpose | Implementation |
|-----------|---------|----------------|
| **Gradient Ascent** | Maximize channel activation | Update pixels to increase target neuron output |
| **L2 Regularization** | Prevent extreme values | Penalize large pixel values |
| **Total Variation** | Encourage smoothness | Penalize rapid pixel changes |
| **Gaussian Blur** | Remove noise | Periodic blur during optimization |
| **Jitter** | Translation invariance | Random spatial shifts |
| **Diverse Init** | Find multiple patterns | Different random seeds |
| **DeepDream** | Amplify patterns | Apply to real images |

## Files Created

### 1. `unet_feature_visualization.py` (23 KB)

**Main implementation** - Complete feature visualization engine for U-Net

**Key classes**:
- `FeatureVisualizer`: Core optimization engine
  - `visualize_channel()`: Generate single visualization
  - `visualize_layer_diverse()`: Multiple channels with diversity
  - `deepdream()`: Amplify patterns in real images

**Features**:
- ✅ Gradient ascent optimization with Adam
- ✅ Multiple regularization techniques
- ✅ Diverse visualizations per channel
- ✅ Multi-octave DeepDream
- ✅ Activation history tracking
- ✅ Grid visualization generation

### 2. `pbs_feature_visualization.sh` (4.7 KB)

**PBS submission script** for HPC execution

**Configuration**:
- Walltime: 4 hours (sufficient for full analysis)
- GPU: 1 × NVIDIA A40
- Memory: 32 GB
- Default: 6 layers × 12 channels × 3 diverse = 216 visualizations

**Included layers**:
- `encoder_1_conv2` - Basic edge/pattern detectors
- `encoder_2_conv2` - Texture combiners
- `encoder_3_conv2` - Complex patterns
- `bottleneck_conv2` - Highest-level features
- `decoder_3_conv2` - Reconstruction patterns
- `decoder_1_conv2` - Final output features

### 3. `FEATURE_VISUALIZATION_GUIDE.md` (17 KB)

**Complete documentation** - Theory, implementation, interpretation

**Contents**:
- Distill article summary
- Technique explanations
- U-Net architecture breakdown
- Usage instructions
- Interpretation guidelines
- Troubleshooting guide

## Quick Start

### Step 1: Submit Job

```bash
cd /scratch/phyzxi/unet-HPC
qsub pbs_feature_visualization.sh
```

### Step 2: Monitor Progress

```bash
# Check job status
qstat -u phyzxi

# Watch live output
tail -f UNet_Feature_Viz.o<jobid>
```

**Expected output**:
```
Visualizing encoder_1_conv2 - Channel 0
Optimizing: 100%|██████████| 500/500 [00:12<00:00, 40.23it/s]

Visualizing encoder_1_conv2 - Channel 1
Optimizing: 100%|██████████| 500/500 [00:11<00:00, 42.15it/s]

...

✓ Saved grid: encoder_1_conv2_diverse_visualizations.png
```

### Step 3: Download Results

From your local machine:

```bash
scp -r <username>@hpc:/path/to/unet_feature_viz_YYYYMMDD_HHMMSS ~/Downloads/
```

### Step 4: View Visualizations

Navigate to downloaded directory and open:
- `<layer>_diverse_visualizations.png` - Grid overview
- `<layer>/ch<###>_div<#>.png` - Individual channels
- `deepdream/*.png` - Enhanced real images

## What You'll See

### Encoder_1 (Early Features)

**Expected patterns**:
- **Edges**: Horizontal, vertical, diagonal lines
- **Gradients**: Brightness transitions
- **Simple textures**: Dots, grids, basic patterns

**Microbead context**:
- Circular edge segments
- Bright-to-dark transitions (particle boundaries)
- Intensity gradients

### Encoder_3 (Mid-Level Features)

**Expected patterns**:
- **Complex textures**: Repeating motifs
- **Combined edges**: Corners, T-junctions
- **Pattern combinations**: Multiple edge orientations

**Microbead context**:
- Partial circular patterns
- Particle spacing patterns
- Texture of particle clusters

### Bottleneck (High-Level Concepts)

**Expected patterns**:
- **Abstract features**: Hard to describe verbally
- **Multi-scale patterns**: Combinations of everything below
- **Semantic concepts**: Task-relevant features

**Microbead context**:
- Full particle representations
- Density patterns
- Segmentation-relevant features

### Decoder_1 (Reconstruction Features)

**Expected patterns**:
- **Boundary refinement**: Edge sharpening
- **Spatial templates**: How to draw outputs
- **Detail recovery**: Fine-grained patterns

**Microbead context**:
- Segmentation mask templates
- Boundary drawing strategies
- Multi-particle separation patterns

## Example Interpretation

### Scenario 1: Clear Circular Patterns

**Observation**: Encoder_3 Channel 45 shows concentric circles

**Interpretation**:
- ✅ Model learned to detect circular particle boundaries
- ✅ Feature is relevant to microbead segmentation
- ✅ This channel likely important for particle detection

### Scenario 2: Random Noise

**Observation**: Multiple channels produce unstructured noise

**Interpretation**:
- ⚠️ These channels may not be effectively used
- ⚠️ Possible overparameterization
- 💡 Consider reducing model capacity

### Scenario 3: Diverse Examples Show Rotations

**Observation**: Channel produces same pattern at different angles

**Interpretation**:
- ✅ Channel is rotation-invariant
- ✅ Good for detecting features at any orientation
- ✅ Model has learned useful invariances

### Scenario 4: DeepDream Creates Artifacts

**Observation**: DeepDream invents particles where none exist

**Interpretation**:
- ⚠️ Model may be too sensitive to certain patterns
- ⚠️ Risk of false positives
- 💡 May need more diverse training data

## Key Parameters to Adjust

### For Quick Exploration

```bash
python unet_feature_visualization.py \
    --channels_per_layer 4 \
    --diverse_per_channel 2 \
    --iterations 200
```

**Time**: ~15 minutes
**Output**: Quick overview of main patterns

### For Publication-Quality Figures

```bash
python unet_feature_visualization.py \
    --channels_per_layer 16 \
    --diverse_per_channel 5 \
    --iterations 1000 \
    --image_size 1024
```

**Time**: ~4 hours
**Output**: High-resolution, well-optimized visualizations

### For Specific Layer Deep Dive

```bash
python unet_feature_visualization.py \
    --layers bottleneck_conv2 \
    --channels_per_layer 32 \
    --diverse_per_channel 5 \
    --iterations 800
```

**Time**: ~2 hours
**Output**: Comprehensive analysis of single layer

## Complementary Analysis

### Combine with CRP for Full Story

1. **Feature Visualization** → What does Channel 45 detect?
   - Result: "Circular boundaries with radius ~10 pixels"

2. **CRP Analysis** → How does Channel 45 influence outputs?
   - Result: "Strongly connected to Decoder_1 Channels [12, 19, 23]"

3. **Combined Insight** → "Circular boundary detection in Encoder_3 Ch45 propagates through Decoder_1 Ch19 to refine segmentation masks"

### Workflow

```
[1] Feature Viz → Understand WHAT each channel detects
                     ↓
[2] CRP Analysis → Understand HOW channels connect
                     ↓
[3] Interpretation → Complete understanding of model behavior
```

## Expected Runtime

| Configuration | Layers | Channels | Diverse | Total Viz | Time |
|---------------|--------|----------|---------|-----------|------|
| **Quick** | 3 | 4 | 2 | 24 | ~15 min |
| **Default** | 6 | 12 | 3 | 216 | ~1 hour |
| **Comprehensive** | 9 | 16 | 5 | 720 | ~3 hours |
| **Publication** | 6 | 16 | 5 | 480 | ~2 hours |

*Times assume GPU execution (NVIDIA A40)*

## Troubleshooting One-Liners

```bash
# Check if job is running
qstat -u phyzxi | grep Feature_Viz

# See recent output
tail -30 UNet_Feature_Viz.o<jobid>

# Count generated visualizations
ls unet_feature_viz_*/encoder_1_conv2/*.png | wc -l

# Check GPU usage during run
ssh <compute-node> nvidia-smi

# Find output directory
ls -td unet_feature_viz_* | head -1

# Quick preview of grid image (if on local machine with display)
open unet_feature_viz_*/encoder_1_conv2_diverse_visualizations.png
```

## Next Steps After Results

1. **✅ Verify model learns meaningful features**
   - Check early layers show edge detectors
   - Check deep layers show task-relevant patterns

2. **🔍 Identify important channels**
   - Which channels show clearest patterns?
   - Which are most interpretable?

3. **🔗 Connect to CRP results**
   - Do important channels (from CRP) show meaningful features?
   - Can you trace feature → connection → output?

4. **📊 Generate publication figures**
   - Grid visualizations for paper figures
   - DeepDream for showing model perception
   - Combine with performance metrics

5. **🔧 Guide model improvements**
   - Dead channels → Reduce model size
   - Missing features → Augment training data
   - Unexpected patterns → Investigate biases

## Files Summary

| File | Size | Purpose |
|------|------|---------|
| `unet_feature_visualization.py` | 23 KB | Main implementation |
| `pbs_feature_visualization.sh` | 4.7 KB | HPC submission |
| `FEATURE_VISUALIZATION_GUIDE.md` | 17 KB | Complete documentation |
| `FEATURE_VIZ_QUICK_START.md` | This file | Quick reference |

**Status**: ✅ Ready to run
**Command**: `qsub pbs_feature_visualization.sh`

---

**Questions answered by this analysis**:
- ❓ What patterns does my U-Net look for?
- ❓ Are learned features interpretable?
- ❓ Does the model focus on meaningful regions?
- ❓ How do different layers represent microbeads?
- ❓ What reconstruction strategies does decoder use?

**Discover your model's internal representations!** 🔬✨
