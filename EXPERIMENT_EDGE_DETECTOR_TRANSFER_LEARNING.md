# Experiment: Edge Detector Transfer Learning in U-Net Layer 1

**Date**: October 30, 2025
**Motivation**: Investigation of why encoder layer 1 learned textures instead of edge detectors
**Hypothesis Test**: Do edge detectors help or hinder cell counting performance?

---

## Research Question

**Observation** (from [FEATURE_VISUALIZATION_COMPARISON_ANALYSIS.md:343](FEATURE_VISUALIZATION_COMPARISON_ANALYSIS.md#L343)):
> ⚠️ **Surprisingly few edge detectors** - Expected Gabor-like filters, got textures

**Question**: Is the absence of edge detectors in layer 1:
1. **A bug**: Random initialization failed to discover useful edge detectors?
2. **A feature**: Task-specific texture features are actually better for cell counting?

---

## Experimental Design

### Control Group (Baseline)
- **Model**: Current U-Net (already trained)
- **Layer 1**: Randomly initialized → learned textures
- **Performance**: IoU = 0.6377

### Treatment A: Frozen Edge Detectors
- **Model**: U-Net with Gabor filter initialization
- **Layer 1**: **FROZEN** (edge detectors stay fixed)
- **Hypothesis**: If edge detectors are sufficient, network learns to use them
- **Expected outcome if hypothesis true**: IoU ≥ 0.63

### Treatment B: Trainable Edge Detectors
- **Model**: U-Net with Gabor filter initialization
- **Layer 1**: **TRAINABLE** (edge detectors can adapt)
- **Hypothesis**: If texture features are better, network adapts edges → textures
- **Expected outcome if hypothesis true**: Layer 1 evolves toward texture features

`★ Experimental Logic ────────────────────────────────────────────────`
**Possible Outcomes:**

1. **Frozen ≈ Trainable ≈ Baseline** → Edge detectors are neutral (neither help nor hurt)
2. **Frozen > Baseline & Trainable > Frozen** → Edge detectors provide good initialization
3. **Frozen < Baseline & Trainable ≈ Baseline** → Edge detectors are sub-optimal; network must adapt them
4. **Frozen << Baseline & Trainable ≈ Baseline** → Edge detectors actively hurt performance

The comparison reveals WHY the network learned textures!
`──────────────────────────────────────────────────────────────────────`

---

## Technical Implementation

### 1. Edge Detector Initialization

**Option 1: Gabor Filters** (Classic edge/orientation detectors)
```python
def create_gabor_filters(n_filters=32, kernel_size=3):
    """
    Create Gabor filters for edge detection at multiple orientations

    Properties:
    - Orientations: 0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5° (8 angles)
    - Frequencies: 2-3 different spatial frequencies
    - Phases: 0° and 90° (for edge and line detection)

    Returns:
        weights: [n_filters, 1, kernel_size, kernel_size]
    """
    filters = []

    # Parameters for diversity
    orientations = np.linspace(0, np.pi, 8)  # 8 orientations
    frequencies = [0.1, 0.2, 0.3]  # 3 spatial frequencies
    phases = [0, np.pi/2]  # Edge and line detectors

    for theta in orientations:
        for freq in frequencies:
            for phi in phases:
                gabor_kernel = create_gabor_kernel(
                    size=kernel_size,
                    theta=theta,    # Orientation
                    freq=freq,      # Spatial frequency
                    phi=phi,        # Phase (0 = edge, π/2 = line)
                    sigma=1.0       # Gaussian envelope width
                )
                filters.append(gabor_kernel)

                if len(filters) >= n_filters:
                    break
            if len(filters) >= n_filters:
                break
        if len(filters) >= n_filters:
            break

    return torch.tensor(filters).unsqueeze(1)  # [n_filters, 1, H, W]
```

**Option 2: ImageNet Pre-trained** (VGG/ResNet layer 1)
```python
def load_imagenet_edges(n_filters=32):
    """
    Extract first conv layer from ImageNet-pretrained VGG16

    Problem: VGG expects RGB (3 channels), we have grayscale (1 channel)
    Solution: Average across RGB channels to get grayscale-compatible filters
    """
    vgg16 = torchvision.models.vgg16(pretrained=True)
    imagenet_weights = vgg16.features[0].weight.data  # [64, 3, 3, 3]

    # Average RGB channels to get grayscale filters
    grayscale_weights = imagenet_weights.mean(dim=1, keepdim=True)  # [64, 1, 3, 3]

    # Select first n_filters
    return grayscale_weights[:n_filters]
```

**Recommendation**: **Use Gabor filters** because:
1. ✅ Explicitly designed for edge/orientation detection
2. ✅ No domain shift (VGG trained on natural images, not microscopy)
3. ✅ Interpretable parameters (orientation, frequency, phase)
4. ✅ Can match 3×3 kernel size exactly

---

### 2. Model Modifications

#### Freezing Layer 1

```python
class UNetWithFreezableLayer1(nn.Module):
    def __init__(self, n_filters=32, dropout=0.2, freeze_layer1=False,
                 init_mode='gabor'):
        super().__init__()

        # Encoder 1 (layer to be frozen/trainable)
        self.enc1 = ConvBlock(1, n_filters, dropout=dropout)

        # Initialize with edge detectors
        if init_mode == 'gabor':
            gabor_weights = create_gabor_filters(n_filters=n_filters, kernel_size=3)
            self.enc1.conv1.weight.data = gabor_weights
        elif init_mode == 'imagenet':
            imagenet_weights = load_imagenet_edges(n_filters=n_filters)
            self.enc1.conv1.weight.data = imagenet_weights
        # else: random initialization (baseline)

        # Freeze layer 1 if requested
        if freeze_layer1:
            for param in self.enc1.parameters():
                param.requires_grad = False

        # Rest of U-Net unchanged
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(n_filters, n_filters*2, dropout=dropout)
        # ... rest of architecture
```

#### Training Script Modifications

```python
def train_with_optional_frozen_layer(freeze_layer1=False, **kwargs):
    """
    Train U-Net with optionally frozen layer 1

    Key changes:
    1. Model initialization with edge detectors
    2. Layer 1 freezing before optimizer creation
    3. Separate tracking of layer 1 weights (for visualization)
    4. Feature visualization after training
    """

    # Create model with edge detector initialization
    model = UNetWithFreezableLayer1(
        n_filters=32,
        dropout=0.2,
        freeze_layer1=freeze_layer1,
        init_mode='gabor'
    )

    # Verify freezing
    if freeze_layer1:
        layer1_params = list(model.enc1.parameters())
        print(f"Layer 1 frozen: {not layer1_params[0].requires_grad}")

    # Optimizer only updates trainable parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=0.001)

    # Training loop unchanged
    for epoch in range(100):
        train_one_epoch(model, train_loader, optimizer, criterion)
        val_iou = validate(model, val_loader)

    # Save model + layer 1 weights for analysis
    torch.save({
        'model_state_dict': model.state_dict(),
        'layer1_frozen': freeze_layer1,
        'layer1_weights': model.enc1.conv1.weight.data.cpu(),
        'val_iou': best_val_iou,
    }, save_path)

    return model, best_val_iou
```

---

### 3. Training Configuration

Use **identical hyperparameters** from best model:

```python
EXPERIMENT_CONFIG = {
    # From best_models_PyTorch/unet/model_info.json
    'n_filters': 32,
    'dropout': 0.2,
    'learning_rate': 0.001,

    # Training settings (same as train_pytorch_comparison_no_aug.py)
    'epochs': 100,
    'batch_size': 4,
    'early_stopping_patience': 20,
    'reduce_lr_patience': 10,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7,

    # Loss function
    'focal_gamma': 2.0,
    'focal_alpha': 0.25,

    # Dataset (same split as original)
    'images_dir': './dataset_shrunk_masks/images/',
    'masks_dir': './dataset_shrunk_masks/masks/',
    'train_val_split': 0.8,
    'random_seed': 42,  # Same seed for reproducible split

    # Experiment-specific
    'init_mode': 'gabor',  # Edge detector type
    'freeze_layer1': [False, True],  # Two versions
}
```

**No hyperparameter search needed** - we're testing layer 1 freezing only!

---

### 4. Feature Visualization

After training both models, generate visualizations using existing infrastructure:

```python
# For each trained model (frozen and trainable)
python unet_feature_visualization.py \
    --model_path ./edge_detector_experiment/unet_frozen_layer1/best_model.pth \
    --layers encoder_1_conv2 encoder_2_conv2 encoder_3_conv2 bottleneck_conv2 decoder_3_conv2 decoder_1_conv2 \
    --n_filters 32 \
    --dropout 0.2 \
    --channels_per_layer 12 \
    --diverse_per_channel 3 \
    --iterations 500
```

**Expected outputs**:
- `unet_frozen_layer1_feature_viz_YYYYMMDD/` - Frozen edge detectors
- `unet_trainable_layer1_feature_viz_YYYYMMDD/` - Adapted edge detectors
- Comparison with `unet_feature_viz_20251029_065244/` - Original (random init)

---

## Analysis Plan

### Performance Comparison

| Model Variant | Layer 1 Init | Layer 1 Status | Val IoU | vs Baseline |
|---------------|-------------|----------------|---------|-------------|
| **Baseline** | Random | Trained | 0.6377 | - |
| **Frozen Edges** | Gabor | Frozen | ? | ? |
| **Trainable Edges** | Gabor | Trained | ? | ? |

**Key Metrics**:
1. **Validation IoU**: Does edge initialization help?
2. **Training convergence**: Does frozen layer slow learning?
3. **Layer 1 weight change**: How much do Gabor filters adapt?

---

### Feature Visualization Comparison

#### Encoder Layer 1 (Three-Way Comparison)

| Variant | Initialization | Training | Expected Features |
|---------|---------------|----------|-------------------|
| **Baseline** | Random | Full | Textures (current observation) |
| **Frozen** | Gabor | Frozen | Pure Gabor filters (unchanged) |
| **Trainable** | Gabor | Full | Gabor → ? (adaptation trajectory) |

**Analysis Questions**:
1. **Do frozen Gabor filters suffice?**
   - If IoU (frozen) ≈ IoU (baseline), edges are sufficient
   - If IoU (frozen) << IoU (baseline), textures are better

2. **Do Gabor filters adapt toward textures?**
   - Compare layer 1 visualizations: Gabor (t=0) vs. Gabor (t=100 epochs)
   - If trainable filters converge to texture-like patterns → textures are optimal
   - If trainable filters stay edge-like → Gabor is good initialization

3. **Do deeper layers compensate?**
   - If frozen layer 1 hurts performance, check if encoder 2/3 learn different features
   - Feature visualization of encoder_2_conv2 for all three models

---

### Quantitative Analysis

#### 1. Layer 1 Weight Evolution

```python
def analyze_layer1_adaptation(initial_weights, final_weights):
    """
    Quantify how much Gabor filters changed during training
    """
    # Weight distance
    l2_distance = torch.norm(final_weights - initial_weights)
    cosine_similarity = F.cosine_similarity(
        initial_weights.flatten(),
        final_weights.flatten(),
        dim=0
    )

    # Orientation analysis
    initial_orientations = compute_dominant_orientations(initial_weights)
    final_orientations = compute_dominant_orientations(final_weights)
    orientation_shift = np.abs(initial_orientations - final_orientations).mean()

    return {
        'l2_distance': l2_distance.item(),
        'cosine_similarity': cosine_similarity.item(),
        'orientation_shift_degrees': np.degrees(orientation_shift),
    }
```

#### 2. Feature Visualization Similarity

```python
def compare_visualizations(baseline_viz, frozen_viz, trainable_viz):
    """
    Compare generated visualizations across three models
    """
    # Texture similarity (SSIM, LPIPS)
    ssim_frozen_vs_baseline = compute_ssim(frozen_viz, baseline_viz)
    ssim_trainable_vs_baseline = compute_ssim(trainable_viz, baseline_viz)

    # Frequency analysis
    freq_baseline = compute_frequency_spectrum(baseline_viz)
    freq_frozen = compute_frequency_spectrum(frozen_viz)
    freq_trainable = compute_frequency_spectrum(trainable_viz)

    return {
        'frozen_resembles_baseline': ssim_frozen_vs_baseline,
        'trainable_resembles_baseline': ssim_trainable_vs_baseline,
        'frozen_is_high_freq': (freq_frozen > freq_baseline).mean(),  # Edges = high freq
        'trainable_is_high_freq': (freq_trainable > freq_baseline).mean(),
    }
```

---

## File Structure

```
edge_detector_experiment_YYYYMMDD/
├── gabor_filters_visualization.png        # Initial Gabor filter set (32 filters)
├── experiment_config.json                 # Hyperparameters + settings
│
├── unet_frozen_layer1/
│   ├── best_model.pth                     # Frozen layer 1 model
│   ├── training_history.csv               # Loss, IoU per epoch
│   ├── model_info.json                    # Hyperparams + IoU
│   └── layer1_weights_epoch000.pth        # Initial Gabor weights
│
├── unet_trainable_layer1/
│   ├── best_model.pth                     # Trainable layer 1 model
│   ├── training_history.csv
│   ├── model_info.json
│   ├── layer1_weights_epoch000.pth        # Initial Gabor weights
│   ├── layer1_weights_epoch050.pth        # Mid-training
│   └── layer1_weights_epoch100.pth        # Final weights (how much changed?)
│
├── unet_frozen_layer1_feature_viz/        # Feature visualizations (frozen)
│   ├── encoder_1_conv2/                   # Should look like Gabor filters
│   ├── encoder_2_conv2/                   # Compensates for frozen layer?
│   └── ...
│
├── unet_trainable_layer1_feature_viz/     # Feature visualizations (trainable)
│   ├── encoder_1_conv2/                   # Adapted Gabor → textures?
│   ├── encoder_2_conv2/
│   └── ...
│
├── analysis/
│   ├── performance_comparison.png         # IoU comparison (3 models)
│   ├── layer1_weight_evolution.png        # Gabor adaptation over time
│   ├── layer1_viz_comparison.png          # Side-by-side: Baseline vs Frozen vs Trainable
│   ├── frequency_analysis.png             # Spectral comparison
│   └── EDGE_DETECTOR_EXPERIMENT_RESULTS.md
│
└── README.md                              # This document
```

---

## Expected Outcomes & Interpretation

### Scenario 1: Frozen Edges Hurt Performance
```
IoU (frozen) = 0.58 << IoU (baseline) = 0.6377
IoU (trainable) = 0.64 ≈ IoU (baseline)
```

**Interpretation**:
- ❌ Edge detectors are **sub-optimal** for cell counting
- ✅ Texture-based features (baseline) are better
- ✅ Network needs freedom to adapt layer 1
- **Conclusion**: Absence of edge detectors is a **feature, not a bug**

**Visualization Evidence**:
- Trainable layer 1 evolves from Gabor → textures (high cosine similarity to baseline)
- Frozen layer 1 stays Gabor-like, but encoder 2/3 must compensate

---

### Scenario 2: Frozen Edges Match or Exceed Baseline
```
IoU (frozen) = 0.64 ≈ IoU (baseline) = 0.6377
IoU (trainable) = 0.65 > IoU (baseline)
```

**Interpretation**:
- ✅ Edge detectors are **sufficient** for cell counting
- ✅ Gabor initialization provides good starting point
- ❓ Random initialization can reach same performance but may take longer
- **Conclusion**: Edge detectors work, but random init also discovers good features

**Visualization Evidence**:
- Frozen layer 1 stays Gabor-like, works fine
- Trainable layer 1 may refine Gabor filters (small adaptation)

---

### Scenario 3: Frozen Edges Dramatically Improve Performance
```
IoU (frozen) = 0.68 >> IoU (baseline) = 0.6377
IoU (trainable) = 0.69 > IoU (frozen)
```

**Interpretation**:
- ✅✅ Edge detectors are **critical** for good performance
- ❌ Random initialization failed to discover them
- ✅ Transfer learning from Gabor filters is highly effective
- **Conclusion**: Absence of edge detectors is a **bug** - we should always pre-train layer 1

**Visualization Evidence**:
- Trainable layer 1 keeps edge-like structure (low adaptation)
- Frozen layer 1 shows organized oriented filters

**Action**: Use Gabor initialization for all future models!

---

### Scenario 4: Complex Interaction
```
IoU (frozen) = 0.61 < IoU (baseline) = 0.6377 < IoU (trainable) = 0.66
```

**Interpretation**:
- ⚠️ Edge detectors provide **good initialization** but shouldn't be frozen
- ✅ Gabor → texture adaptation improves over random init
- ✅ Transfer learning helps, but task-specific adaptation needed
- **Conclusion**: Use Gabor initialization, but allow training (warm start)

**Visualization Evidence**:
- Trainable layer 1 shows hybrid features (edges + textures)
- Initial Gabor structure biases learning toward better features

---

## Timeline

| Day | Task | Output |
|-----|------|--------|
| **Day 1** | Create Gabor filter initialization module | `gabor_initializer.py` |
| **Day 1** | Create training script with freezing capability | `train_edge_detector_experiment.py` |
| **Day 1** | Create PBS submission scripts | `pbs_frozen_layer1.sh`, `pbs_trainable_layer1.sh` |
| **Day 2-3** | Run training (frozen version) on HPC | ~24 hours |
| **Day 2-3** | Run training (trainable version) on HPC | ~24 hours (parallel) |
| **Day 4** | Generate feature visualizations | ~6 hours total |
| **Day 5** | Analysis and comparison | `EDGE_DETECTOR_EXPERIMENT_RESULTS.md` |

**Total time**: ~5 days (2-3 days if both training jobs run in parallel)

---

## Success Criteria

✅ **Experiment succeeds if**:
1. Both training jobs complete without errors
2. Frozen layer 1 demonstrably stays unchanged (weight comparison)
3. Feature visualizations generated for both models
4. Clear performance comparison (IoU difference > 0.01)
5. Visual evidence of Gabor adaptation (or lack thereof)

✅ **Scientific question answered if**:
- We can confidently say whether texture features are optimal for this task
- We understand if random initialization was sufficient or sub-optimal
- We have actionable guidance for future model training

---

## Next Steps

1. **Create implementation files** (ready to execute)
2. **Submit HPC jobs** (parallel training)
3. **Monitor training progress** (layer 1 weight tracking)
4. **Generate visualizations** (3-way comparison)
5. **Publish findings** (update FEATURE_VISUALIZATION_COMPARISON_ANALYSIS.md)

---

## References

1. [FEATURE_VISUALIZATION_COMPARISON_ANALYSIS.md](FEATURE_VISUALIZATION_COMPARISON_ANALYSIS.md) - Original observation
2. [best_models_PyTorch/unet/model_info.json](best_models_PyTorch/unet/model_info.json) - Baseline hyperparameters
3. [train_pytorch_comparison_no_aug.py](train_pytorch_comparison_no_aug.py) - Training infrastructure
4. [unet_feature_visualization.py](unet_feature_visualization.py) - Visualization infrastructure

---

**Summary**: This experiment will definitively answer whether the absence of edge detectors in layer 1 is a shortcoming of random initialization or a valid adaptation to the cell counting task. The controlled comparison (frozen vs. trainable vs. baseline) isolates the effect of layer 1 initialization on overall performance. 🔬
