# Claude Code Project Configuration

## Project Overview

**Extended Mitochondria Segmentation Study: Classical U-Net to Modern Hybrid Architectures**

**Phase 1 (Completed):** U-Net, Attention U-Net, Attention ResU-Net
**Phase 2 (Proposed):** ConvNeXt-UNet, Swin-UNet, CoAtNet-UNet

## Scientific Motivation

### Research Questions
1. **Data Efficiency**: Performance with limited medical imaging data (~1980 patches)?
2. **Segmentation Quality**: Do modern architectures improve Jaccard/Dice over classical U-Net?
3. **Architecture Trade-offs**: Performance-complexity-speed balance?
4. **Inductive Bias**: Strong conv bias (ConvNeXt) vs weak bias (Swin) in medical imaging?

### Architecture Rationale
- **ConvNeXt**: Modernized CNN with transformer principles (7×7 kernels, LayerNorm, inverted bottleneck)
- **Swin Transformer**: Hierarchical shifted-window attention for multi-scale features
- **CoAtNet**: Hybrid C-C-T-T stacking balancing generalization and capacity

## Dataset Configuration

- **Source**: EPFL mitochondria EM dataset
- **Patches**: 1,980 images at 256×256 resolution
- **Split**: 90% train / 10% validation
- **Loss**: Binary Focal Loss (γ=2)
- **Metrics**: Jaccard, Dice, accuracy
- **Optimizer**: Adam (lr=1e-3), batch=8, epochs=100

## Expected Performance (Hypothesis)

1. **CoAtNet-UNet**: ~85-90% Jaccard (best overall - hybrid balance)
2. **ConvNeXt-UNet**: ~82-87% Jaccard (best data efficiency)
3. **Swin-UNet**: ~80-85% Jaccard (high capacity, may overfit)
4. **Baselines**: U-Net ~78-82%, Att-UNet ~80-84%, Att-ResUNet ~81-85%

## Proposed Directory Structure