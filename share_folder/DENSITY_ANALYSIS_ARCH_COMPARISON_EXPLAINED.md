# Density Analysis Architecture Comparison - Complete Workflow

## 📂 Directory: `density_analysis_arch_comparison_20251014_004358`

This document explains which code generated this analysis and which models were used for predictions.

---

## 🎯 Quick Answer

### **Scripts Used:**
1. **Shell Script**: [pbs_density_analysis.sh](../density_analysis_arch_comparison_20251014_004358/pbs_density_analysis.sh)
   - PBS job submission script for HPC cluster
   - Orchestrates the entire workflow

2. **Python Script**: [density_analysis_arch_comparison.py](../density_analysis_arch_comparison_20251014_004358/density_analysis_arch_comparison.py)
   - Main analysis script
   - Trains models, generates predictions, creates visualizations

### **Models Used:**
⚠️ **Models were TRAINED from scratch, not pre-existing!**

- **UNet** - Trained fresh during analysis
- **ResUNet** - Trained fresh during analysis
- **Attention ResUNet** - Trained fresh during analysis
- **CLAHE+OTSU** - Traditional CV method (no training)

**Framework:** Keras/TensorFlow

---

## 📋 Complete Workflow

### **Stage 1: Job Submission (pbs_density_analysis.sh)**

**Purpose:** Submit analysis job to HPC cluster

**Key Actions:**
```bash
#!/bin/bash
#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=36:ngpus=1:mem=240gb

cd /home/svu/phyzxi/scratch/unet-HPC
module load singularity
image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

singularity exec --nv "$image" python3 density_analysis_arch_comparison.py
```

**What it does:**
1. Requests HPC resources (1 GPU, 36 CPUs, 240GB RAM, 8 hours)
2. Loads TensorFlow 2.16.1 Singularity container
3. Executes Python script with GPU support (`--nv` flag)
4. Logs output to `density_analysis_console_20251014_084334.log`

---

### **Stage 2: Model Training & Prediction (density_analysis_arch_comparison.py)**

**Purpose:** Train models and perform density analysis

#### **Configuration (Lines 53-76)**

```python
CONFIG = {
    # Directories
    'dataset_dir': './dataset_full_stack',        # Training data
    'test_images_dir': './test_images',          # Test images (dilution series)
    'output_dir': './density_analysis_arch_comparison_20251014_004358',

    # Model configuration (from validation_arch_comparison)
    'architectures': ['unet', 'resunet', 'attention_resunet'],
    'img_size': 256,
    'img_channels': 1,              # Grayscale
    'filters': 64,                  # Base number of filters
    'dropout': 0.2,
    'batch_size': 16,
    'learning_rate': 5e-5,
    'epochs': 50,
    'early_stopping_patience': 10,

    # CLAHE+OTSU parameters
    'clahe': {'clipLimit': 2.0, 'tileGridSize': (8, 8)},
}
```

---

#### **Step 1: Load Training Data (Lines 144-180)**

```python
def load_training_data(dataset_dir, img_size=256):
    """Load training data from dataset_full_stack."""
    images_dir = Path(dataset_dir) / 'images'
    masks_dir = Path(dataset_dir) / 'masks'

    # Load 1980 image-mask pairs
    # Resize to 256×256
    # Convert to grayscale
    # Normalize to [0, 1]

    return X, y  # Shape: (1980, 256, 256, 1)
```

**Output:**
```
Loading 1980 training images...
✓ Loaded 1980 image-mask pairs
Train: 1584, Val: 396  (80/20 split)
```

---

#### **Step 2: Train Models (Lines 208-280)**

```python
def train_model(architecture, X_train, y_train, X_val, y_val, output_dir):
    """Train a single architecture."""

    # Build model using model_architectures.py
    model = get_model(
        model_name=architecture,
        input_shape=(256, 256, 1),
        NUM_CLASSES=1,
        dropout_rate=0.2,
        batch_norm=True
    )

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        loss=combined_dice_focal_loss,
        metrics=[jacard_coef, dice_coef]
    )

    # Train with callbacks
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[ModelCheckpoint, EarlyStopping, ReduceLROnPlateau]
    )

    # Save best model
    model.save(f'{architecture}_best_model.keras')

    return model, best_jaccard
```

**Models Trained:**

1. **UNet** ([lines 506-511](../density_analysis_arch_comparison_20251014_004358/density_analysis_arch_comparison.py#L506-L511))
   - Architecture: Standard encoder-decoder with skip connections
   - Saved to: `trained_models/unet_best_model.keras`

2. **ResUNet**
   - Architecture: UNet with residual connections
   - Saved to: `trained_models/resunet_best_model.keras`

3. **Attention ResUNet**
   - Architecture: ResUNet with attention gates
   - Saved to: `trained_models/attention_resunet_best_model.keras`

**Training Parameters:**
- **Loss**: Combined Dice + Focal Loss
- **Optimizer**: Adam (lr=5e-5)
- **Epochs**: Up to 50 (with early stopping)
- **Batch size**: 16
- **Image size**: 256×256 grayscale
- **Filters**: 64 (base)
- **Dropout**: 0.2

---

#### **Step 3: Load Test Images (Lines 182-202)**

```python
def load_test_images(test_dir):
    """Load test images from dilution series."""
    image_files = list(Path(test_dir).glob('*.tif'))
    return image_files
```

**Test Images:** Dilution series (10x, 20x, 80x, ..., 10240x)

---

#### **Step 4: Predict on Test Images (Lines 520-565)**

```python
all_data = []

for img_path in test_images:
    dilution = extract_dilution_factor(img_path.stem)  # e.g., 10, 20, 80

    # Predict with each DL model
    for arch in ['unet', 'resunet', 'attention_resunet']:
        densities = predict_on_image(trained_models[arch], img_path)

        for density in densities:
            all_data.append({
                'image': img_path.stem,
                'dilution_factor': dilution,
                'method': arch,
                'foreground_pct': density
            })

    # Apply CLAHE+OTSU (traditional method)
    densities = clahe_otsu_on_image(img_path)

    for density in densities:
        all_data.append({
            'image': img_path.stem,
            'dilution_factor': dilution,
            'method': 'clahe_otsu',
            'foreground_pct': density
        })
```

**What happens:**
1. Load each test image
2. Extract dilution factor from filename (e.g., "image_160x.tif" → 160)
3. For each DL model:
   - Split image into 256×256 tiles
   - Predict mask for each tile
   - Calculate foreground percentage (density)
4. Apply CLAHE+OTSU as baseline comparison
5. Collect all density measurements

---

#### **Step 5: Generate Visualizations (Lines 400-470)**

```python
def plot_density_vs_dilution_separate(df, method, output_path):
    """Create individual density plot for each method."""

    # Filter data for this method
    df_method = df[df['method'] == method]

    # Create boxplot: dilution factor vs foreground percentage
    # Log scale on Y-axis
    # Viridis-like color scheme

    plt.savefig(output_path, dpi=300)
```

**Generated Plots:**

1. `plots/unet_density_vs_dilution_CORRECTED.png`
2. `plots/resunet_density_vs_dilution_CORRECTED.png`
3. `plots/attention_resunet_density_vs_dilution_CORRECTED.png`
4. `plots/clahe_otsu_density_vs_dilution_CORRECTED.png`

**CSV Data:**

`csv_data/density_analysis_comprehensive.csv` - All density measurements with columns:
- `image`: Image filename
- `dilution_factor`: Dilution level (10, 20, 80, ...)
- `method`: Model/method used (unet, resunet, attention_resunet, clahe_otsu)
- `foreground_pct`: Calculated density

---

## 🔍 Key Insights

`★ Insight ─────────────────────────────────────`
**Why Train Models Fresh Instead of Using Pre-existing Models?**

1. **Controlled Comparison**: All models trained with identical:
   - Training data (dataset_full_stack)
   - Hyperparameters (filters=64, dropout=0.2, lr=5e-5)
   - Loss function (Combined Dice + Focal)
   - Training procedure (same callbacks, epochs)

2. **Fair Evaluation**: Ensures architectural differences are the only variable

3. **Configuration Matching**: Uses same config as `validation_arch_comparison_20251013_093844`:
   - Filters: 64
   - Dropout: 0.2
   - Learning rate: 5e-5
   - Batch size: 16

4. **Self-Contained Workflow**: One script does everything (train → predict → analyze)
`─────────────────────────────────────────────────`

---

## 📊 Model Architecture Details

### **1. UNet (Baseline)**

```
Input (256×256×1)
    ↓
Encoder (4 levels, 64→128→256→512 filters)
    → MaxPool after each level
    ↓
Bottleneck (1024 filters)
    ↓
Decoder (4 levels, 512→256→128→64 filters)
    → Transpose Conv + Skip Connection
    ↓
Output (256×256×1, Sigmoid)
```

**Defined in:** `model_architectures.py`

---

### **2. ResUNet (Residual Connections)**

```
Same as UNet but:
- ConvBlock → ResidualBlock
- Residual shortcuts: x + F(x)
- Better gradient flow
```

**Advantage:** Deeper network without vanishing gradients

---

### **3. Attention ResUNet (Residual + Attention)**

```
Same as ResUNet but:
- Attention Gates before skip connections
- Filters irrelevant features from encoder
- Focuses on foreground (beads)
```

**Advantage:** Better at ignoring background noise

---

### **4. CLAHE+OTSU (Traditional CV)**

```
Input Image
    ↓
CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clipLimit=2.0, tileGridSize=(8,8)
    ↓
OTSU Thresholding (automatic threshold selection)
    ↓
Binary Mask
```

**Defined in:** [density_analysis_arch_comparison.py:294-298](../density_analysis_arch_comparison_20251014_004358/density_analysis_arch_comparison.py#L294-L298)

**Advantage:** No training required, fast, interpretable

---

## 🎯 Purpose of This Analysis

### **Research Question:**

> "How do different UNet architectures compare to traditional CV methods for bead density estimation across dilution factors?"

### **Hypothesis:**

Deep learning models should:
1. Match or exceed CLAHE+OTSU baseline
2. Maintain accuracy across dilution range (10x to 10240x)
3. Show architectural differences (attention > residual > vanilla)

---

## 🛠️ How to Reproduce

### **Option 1: Run on HPC Cluster**

```bash
# Submit PBS job
qsub pbs_density_analysis.sh

# Check job status
qstat

# Monitor output log (tail -f or less)
tail -f density_analysis_console_TIMESTAMP.log
```

---

### **Option 2: Run Locally (if you have GPU)**

```bash
# Make sure you have TensorFlow, Keras, cv2, etc.
python3 density_analysis_arch_comparison.py
```

**Requirements:**
- GPU with CUDA support (for reasonable training time)
- TensorFlow >= 2.10
- ~240GB RAM (for loading 1980 training images)
- `./dataset_full_stack/` with training data
- `./test_images/` with dilution series
- `model_architectures.py` (defines UNet, ResUNet, Attention ResUNet)
- `loss_functions_fixed.py` (defines combined_dice_focal_loss)

---

## 📈 Expected Runtime

**On HPC (1× NVIDIA A40 GPU):**
- Training (3 models): ~3-4 hours
- Prediction & Analysis: ~1-2 hours
- **Total**: ~4-6 hours

**On CPU only:**
- Training: ~20-30 hours (NOT recommended!)

---

## 📄 Output Files

```
density_analysis_arch_comparison_20251014_004358/
├── pbs_density_analysis.sh                        # Job script (copied)
├── density_analysis_arch_comparison.py            # Analysis script (copied)
├── density_analysis_console_20251014_084334.log   # Execution log
│
├── trained_models/                                # MISSING (not saved)
│   ├── unet_best_model.keras
│   ├── resunet_best_model.keras
│   └── attention_resunet_best_model.keras
│
├── plots/
│   ├── unet_density_vs_dilution_CORRECTED.png
│   ├── resunet_density_vs_dilution_CORRECTED.png
│   ├── attention_resunet_density_vs_dilution_CORRECTED.png
│   └── clahe_otsu_density_vs_dilution_CORRECTED.png
│
└── csv_data/
    └── density_analysis_comprehensive.csv         # All density measurements
```

**Note:** Trained models were NOT saved in the output directory (only stored temporarily).

---

## 🔧 Key Configuration Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Image Size** | 256×256 | Balance between detail and memory |
| **Channels** | 1 (grayscale) | Beads are monochrome |
| **Filters** | 64 | From validation_arch_comparison |
| **Dropout** | 0.2 | Prevents overfitting |
| **Learning Rate** | 5e-5 | Small LR for stable training |
| **Batch Size** | 16 | Fits in GPU memory |
| **Loss** | Dice + Focal | Handles class imbalance |
| **Epochs** | 50 (early stop) | Prevents overfitting |

---

## 🆚 Comparison with PyTorch Pipeline

| Aspect | This Analysis | PyTorch Pipeline |
|--------|---------------|------------------|
| **Framework** | Keras/TensorFlow | PyTorch |
| **Purpose** | Density analysis | Hyperparameter search |
| **Models** | Trained fresh | Pre-trained from experiments |
| **Image Size** | 256×256 | 512×512 |
| **Channels** | 1 (grayscale) | 1 (grayscale) or 3 (RGB) |
| **Training Data** | dataset_full_stack | dataset_shrunk_masks |
| **Test Data** | Dilution series | Same |
| **Output** | Density plots + CSV | Predictions + Density analysis |
| **Date** | Oct 14, 2025 | Oct 21-22, 2025 |

---

## 📝 Summary

### **Scripts:**
1. **[pbs_density_analysis.sh](../density_analysis_arch_comparison_20251014_004358/pbs_density_analysis.sh)** - HPC job submission
2. **[density_analysis_arch_comparison.py](../density_analysis_arch_comparison_20251014_004358/density_analysis_arch_comparison.py)** - Main analysis

### **Models:**
- **UNet, ResUNet, Attention ResUNet** - Trained fresh from `dataset_full_stack`
- **CLAHE+OTSU** - Traditional CV baseline (no training)

### **Workflow:**
```
Load Training Data (1980 images)
    ↓
Train 3 Models (UNet, ResUNet, Attention ResUNet)
    ↓
Load Test Images (dilution series)
    ↓
Predict Masks (each model + CLAHE+OTSU)
    ↓
Calculate Density (foreground percentage per tile)
    ↓
Generate Plots & CSV
```

### **Framework:**
**Keras/TensorFlow** (NOT PyTorch)

---

**Author:** Claude Code
**Date:** October 25, 2025
**Purpose:** Document density analysis workflow

---

**End of Documentation**
