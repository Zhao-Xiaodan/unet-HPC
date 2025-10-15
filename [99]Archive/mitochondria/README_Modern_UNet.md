# Modern U-Net Architectures for Mitochondria Segmentation

This directory contains implementations of state-of-the-art U-Net architectures for mitochondria segmentation, extending the existing project with modern deep learning techniques.

## 🚀 New Architectures Implemented

### 1. ConvNeXt-UNet
- **Based on:** ConvNeXt (A ConvNet for the 2020s) - [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- **Key Features:**
  - Depthwise separable convolutions
  - Layer normalization
  - GELU activation
  - Layer scale parameters
  - Improved efficiency over traditional CNNs

### 2. Swin-UNet
- **Based on:** Swin Transformer - [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- **Key Features:**
  - Hierarchical vision transformer
  - Shifted window attention mechanism
  - Multi-scale feature extraction
  - Linear computational complexity

### 3. CoAtNet-UNet
- **Based on:** CoAtNet (Marrying Convolution and Attention) - [arXiv:2106.04803](https://arxiv.org/abs/2106.04803)
- **Key Features:**
  - Hybrid convolution + attention design
  - Progressive attention integration
  - Best of both CNN and Transformer worlds
  - Efficient multi-scale processing

## 📁 File Structure

```
├── modern_unet_models.py      # Model architectures implementation
├── modern_unet_training.py    # Training script for modern architectures
├── pbs_modern_unet.sh        # HPC PBS job submission script
├── test_modern_unet.py       # Validation and testing script
└── README_Modern_UNet.md     # This documentation
```

## 🛠️ Installation and Setup

### Prerequisites
- TensorFlow 2.16.1+ with CUDA support
- Python 3.8+
- Required dependencies from original project
- Access to HPC cluster with Singularity support

### Dataset Requirements
The modern architectures are designed for the same dataset structure as the original project:

**Preferred (Full Stack Dataset - 1,980 patches):**
```
dataset_full_stack/
├── images/          # 1,980 .tif image files
└── masks/           # 1,980 .tif mask files
```

**Alternative (Standard Dataset):**
```
dataset/
├── images/          # .tif image files
└── masks/           # .tif mask files
```

## 🚀 Usage

### Method 1: HPC Cluster Training (Recommended)

1. **Prepare the environment:**
   ```bash
   # Copy all files to your HPC scratch directory
   cp modern_unet_models.py /home/svu/phyzxi/scratch/unet-HPC/
   cp modern_unet_training.py /home/svu/phyzxi/scratch/unet-HPC/
   cp pbs_modern_unet.sh /home/svu/phyzxi/scratch/unet-HPC/
   cp test_modern_unet.py /home/svu/phyzxi/scratch/unet-HPC/
   ```

2. **Ensure dataset is available:**
   ```bash
   # Check dataset structure
   ls -la dataset_full_stack/images/ | wc -l  # Should show ~1,980 files
   ls -la dataset_full_stack/masks/ | wc -l   # Should show ~1,980 files
   ```

3. **Submit the job:**
   ```bash
   qsub pbs_modern_unet.sh
   ```

4. **Monitor progress:**
   ```bash
   # Check job status
   qstat -u $USER

   # Monitor output (when job starts)
   tail -f Modern_UNet_Mitochondria_Segmentation.o<JOBID>
   ```

### Method 2: Local Testing and Validation

1. **Activate conda environment:**
   ```bash
   conda activate unetCNN
   ```

2. **Run validation test:**
   ```bash
   python test_modern_unet.py
   ```

3. **Run training locally (if resources allow):**
   ```bash
   python modern_unet_training.py
   ```

## ⚙️ Configuration

### Training Parameters
The modern architectures use optimized parameters:

```python
# Training Configuration
learning_rate = 1e-4     # Conservative for stability
batch_size = 4           # Smaller due to model complexity
epochs = 100            # With early stopping
optimizer = AdamW       # For transformer models
loss = BinaryFocalLoss  # Same as original project
```

### Model-Specific Optimizations
- **ConvNeXt-UNet:** Uses Adam optimizer with standard learning rate
- **Swin-UNet:** Uses AdamW with weight decay (0.05) for transformer training
- **CoAtNet-UNet:** Uses AdamW with attention-specific regularization

### GPU Memory Requirements
- **ConvNeXt-UNet:** ~8-12 GB GPU memory
- **Swin-UNet:** ~10-16 GB GPU memory
- **CoAtNet-UNet:** ~6-10 GB GPU memory

## 📊 Expected Output

### Generated Files
After successful training, you'll find in the output directory:

```
modern_unet_training_YYYYMMDD_HHMMSS/
├── ConvNeXt_UNet_lr0.0001_bs4_model.hdf5
├── ConvNeXt_UNet_lr0.0001_bs4_history.csv
├── ConvNeXt_UNet_lr0.0001_bs4_results.json
├── Swin_UNet_lr0.0001_bs4_model.hdf5
├── Swin_UNet_lr0.0001_bs4_history.csv
├── Swin_UNet_lr0.0001_bs4_results.json
├── CoAtNet_UNet_lr0.0001_bs4_model.hdf5
├── CoAtNet_UNet_lr0.0001_bs4_history.csv
├── CoAtNet_UNet_lr0.0001_bs4_results.json
├── modern_unet_training_comparison.png
├── modern_unet_performance_summary.png
├── modern_unet_performance_summary.csv
└── training_console_YYYYMMDD_HHMMSS.log
```

### Performance Metrics
Each model generates comprehensive metrics:
- **best_val_jaccard:** Peak validation Jaccard coefficient
- **training_time_seconds:** Total training time
- **model_parameters:** Number of trainable parameters
- **val_loss_stability:** Training stability measure
- **overfitting_gap:** Generalization indicator

## 🔧 Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   ```bash
   # Solution: Reduce batch size in modern_unet_training.py
   batch_size = 2  # Instead of 4
   ```

2. **Dataset Not Found**
   ```bash
   # Ensure proper directory structure
   ls -la dataset_full_stack/images/
   ls -la dataset_full_stack/masks/
   ```

3. **Dependency Issues**
   ```bash
   # Install focal loss if needed
   pip install focal-loss
   ```

4. **Model Creation Errors**
   ```bash
   # Test model creation
   python test_modern_unet.py
   ```

### Performance Optimization

1. **For Limited GPU Memory:**
   - Reduce batch size to 2
   - Use gradient checkpointing (implemented in models)
   - Enable mixed precision training

2. **For Faster Training:**
   - Use the full stack dataset (1,980 patches)
   - Ensure SSD storage for dataset
   - Monitor GPU utilization

## 📈 Performance Comparison

### Expected Performance vs. Original Models

Based on the model architectures, expected improvements:

| Architecture | Expected Jaccard | Training Time | Parameters |
|-------------|------------------|---------------|------------|
| **ConvNeXt-UNet** | 0.94-0.96 | 3-5 hours | ~15-25M |
| **Swin-UNet** | 0.95-0.97 | 4-8 hours | ~20-35M |
| **CoAtNet-UNet** | 0.94-0.96 | 3-6 hours | ~10-20M |

*Original U-Net baseline: ~0.93 Jaccard*

### Key Advantages

1. **Better Feature Extraction:** Modern architectures capture multi-scale features more effectively
2. **Improved Generalization:** Advanced regularization and attention mechanisms
3. **Efficiency:** Better parameter utilization and computational efficiency
4. **Stability:** More stable training dynamics with modern optimizers

## 🔬 Technical Details

### Model Architecture Comparison

| Component | ConvNeXt-UNet | Swin-UNet | CoAtNet-UNet |
|-----------|---------------|-----------|--------------|
| **Encoder** | ConvNeXt blocks | Swin Transformer | Conv + Attention |
| **Attention** | None | Shifted Window | Multi-head |
| **Normalization** | LayerNorm | LayerNorm | BatchNorm + LayerNorm |
| **Activation** | GELU | GELU | GELU |
| **Skip Connections** | Standard | Standard | Enhanced |

### Implementation Features

1. **Memory Optimization:**
   - Gradient checkpointing for large models
   - Progressive memory allocation
   - Automatic batch size adjustment

2. **Training Stability:**
   - Layer-wise learning rate decay
   - Gradient clipping
   - Advanced learning rate scheduling

3. **Monitoring:**
   - Real-time stability tracking
   - Comprehensive metrics logging
   - Automatic visualization generation

## 📚 References

1. **ConvNeXt:** Liu, Z., et al. "A ConvNet for the 2020s." CVPR 2022.
2. **Swin Transformer:** Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
3. **CoAtNet:** Dai, Z., et al. "CoAtNet: Marrying Convolution and Attention for All Data Sizes." NeurIPS 2021.
4. **U-Net:** Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.

## 🤝 Integration with Existing Project

This implementation seamlessly integrates with the existing mitochondria segmentation project:

- **Compatible Metrics:** Uses same Jaccard coefficient and loss functions
- **Dataset Format:** Works with existing dataset structure
- **Output Format:** Generates compatible .hdf5 model files
- **Analysis Tools:** Compatible with existing analysis scripts

### Comparison with Existing Models

To compare with the original U-Net, Attention U-Net, and Attention ResU-Net:

```python
# Load and compare results
import pandas as pd

# Original models results
original_results = pd.read_csv('path/to/original/results.csv')

# Modern models results
modern_results = pd.read_csv('modern_unet_performance_summary.csv')

# Create comprehensive comparison
comparison = compare_all_architectures(original_results, modern_results)
```

## 🎯 Next Steps

1. **Training:** Submit the PBS job and monitor training progress
2. **Analysis:** Compare results with original U-Net architectures
3. **Optimization:** Fine-tune best-performing architecture
4. **Deployment:** Prepare best model for production use
5. **Research:** Consider ensemble methods combining multiple architectures

---

**For questions or issues, please refer to the troubleshooting section or check the console logs in the output directory.**