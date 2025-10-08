#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Microscope_Optimized_UNet
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# OPTIMIZED MITOCHONDRIA SEGMENTATION - MICROSCOPE DATASET
# =======================================================================
# Training UNet, Attention UNet, and Attention ResUNet with optimized
# hyperparameters from the hyperparameter optimization study
#
# Optimized Configurations:
# - UNet: LR=1e-3, Batch Size=8, Expected Val Jaccard ~0.0670
# - Attention UNet: LR=1e-4, Batch Size=16, Expected Val Jaccard ~0.0699
# - Attention ResUNet: LR=5e-4, Batch Size=16, Expected Val Jaccard ~0.0695
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "OPTIMIZED MITOCHONDRIA SEGMENTATION - MICROSCOPE DATASET"
echo "======================================================================="
echo "Dataset: dataset_microscope/"
echo "Models: UNet, Attention UNet, Attention ResUNet"
echo "Using optimized hyperparameters from optimization study"
echo ""

# Job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $(free -h | grep Mem | awk '{print $2}'), CPUs: $(nproc)"
echo ""

# =======================================================================
# ENVIRONMENT SETUP
# =======================================================================

echo "=== TRAINING CONFIGURATION ==="
echo "Dataset: ./dataset_microscope/"
echo "  Images: ./dataset_microscope/images/"
echo "  Masks: ./dataset_microscope/masks/"
echo "Image Size: 256x256"
echo ""
echo "Model Configurations (Optimized):"
echo "  1. UNet:"
echo "     - Learning Rate: 1e-3"
echo "     - Batch Size: 8"
echo "     - Expected Val Jaccard: ~0.0670"
echo ""
echo "  2. Attention UNet (Best Performer):"
echo "     - Learning Rate: 1e-4"
echo "     - Batch Size: 16"
echo "     - Expected Val Jaccard: ~0.0699"
echo ""
echo "  3. Attention ResUNet:"
echo "     - Learning Rate: 5e-4"
echo "     - Batch Size: 16"
echo "     - Expected Val Jaccard: ~0.0695"
echo ""
echo "All models use:"
echo "  - Binary Focal Loss (gamma=2)"
echo "  - Gradient Clipping (clipnorm=1.0)"
echo "  - Early Stopping (patience=15)"
echo "  - Adaptive LR Reduction"
echo "  - Maximum 100 epochs per model"
echo "=============================="
echo ""

# TensorFlow environment variables
export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export CUDA_VISIBLE_DEVICES=0

# Load required modules
module load singularity

# Use the modern TensorFlow container
image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

if [ ! -f "$image" ]; then
    echo "ERROR: TensorFlow container not found at $image"
    echo "Available TensorFlow containers:"
    ls -la /app1/common/singularity-img/hopper/tensorflow/
    exit 1
fi

echo "TensorFlow Container: $image"
echo ""

# =======================================================================
# PRE-EXECUTION CHECKS
# =======================================================================

echo "=== PRE-EXECUTION CHECKS ==="

# Check dataset structure
echo "1. Checking dataset structure..."
if [ -d "./dataset_microscope/images/" ] && [ -d "./dataset_microscope/masks/" ]; then
    echo "   ✓ Dataset directories found"

    # Count files (support multiple image formats)
    img_count=$(find ./dataset_microscope/images/ -type f \( -name "*.tif" -o -name "*.tiff" -o -name "*.png" -o -name "*.jpg" \) | wc -l)
    mask_count=$(find ./dataset_microscope/masks/ -type f \( -name "*.tif" -o -name "*.tiff" -o -name "*.png" -o -name "*.jpg" \) | wc -l)

    echo "   ✓ Images found: $img_count files"
    echo "   ✓ Masks found: $mask_count files"

    if [ $img_count -eq 0 ] || [ $mask_count -eq 0 ]; then
        echo "   ERROR: No image files found in dataset directories"
        echo "   Please ensure your dataset contains image files"
        exit 1
    fi

    if [ $img_count -ne $mask_count ]; then
        echo "   WARNING: Unequal number of images ($img_count) and masks ($mask_count)"
        echo "   This may cause issues during training"
    fi
else
    echo "   ERROR: Dataset directories not found!"
    echo "   Expected: ./dataset_microscope/images/ and ./dataset_microscope/masks/"
    echo "   Current directory contents:"
    ls -la ./
    exit 1
fi

# Check if Python files exist
echo ""
echo "2. Checking Python files..."
if [ -f "./microscope_optimized_training.py" ]; then
    echo "   ✓ Optimized training script found"
else
    echo "   ERROR: Required Python file not found"
    echo "   Expected: microscope_optimized_training.py"
    ls -la ./*.py
    exit 1
fi

if [ -f "./224_225_226_models.py" ] && [ ! -f "./models.py" ]; then
    echo "   ✓ Creating models.py from 224_225_226_models.py"
    cp ./224_225_226_models.py ./models.py
elif [ -f "./models.py" ]; then
    echo "   ✓ models.py found"
else
    echo "   ERROR: Model definitions file not found"
    echo "   Expected: models.py or 224_225_226_models.py"
    exit 1
fi

echo "=========================="
echo ""

# =======================================================================
# TENSORFLOW AND GPU STATUS CHECK
# =======================================================================

echo "=== TENSORFLOW & GPU STATUS ==="
singularity exec --nv "$image" python3 -c "
import tensorflow as tf
import sys

print('Python version:', sys.version.split()[0])
print('TensorFlow version:', tf.__version__)
print('CUDA built support:', tf.test.is_built_with_cuda())

# List GPUs
gpus = tf.config.list_physical_devices('GPU')
print('Physical GPUs found:', len(gpus))
for i, gpu in enumerate(gpus):
    print(f'  GPU {i}: {gpu}')

# Enable memory growth
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('✓ GPU memory growth enabled')
    except RuntimeError as e:
        print('Memory growth error:', e)

# Test basic GPU operation
if gpus:
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([1.0, 2.0])
            b = tf.constant([2.0, 3.0])
            c = tf.add(a, b)
        print('✓ Basic GPU operation successful')
    except Exception as e:
        print('GPU operation failed:', e)

# Check key dependencies
deps_to_check = ['cv2', 'PIL', 'matplotlib', 'numpy', 'sklearn', 'pandas']
print()
print('Checking dependencies:')
for dep in deps_to_check:
    try:
        __import__(dep)
        print(f'  ✓ {dep}')
    except ImportError:
        print(f'  ✗ {dep} - Missing!')

# Special check for focal_loss
try:
    from focal_loss import BinaryFocalLoss
    print('  ✓ focal_loss')
except ImportError:
    print('  ✗ focal_loss - May need installation')
    print('    Will attempt to install during execution')
"
echo "==============================="
echo ""

# =======================================================================
# INSTALL FOCAL LOSS IF NEEDED
# =======================================================================

echo "=== INSTALLING DEPENDENCIES ==="
singularity exec --nv "$image" python3 -c "
import sys
import subprocess

def install_focal_loss():
    try:
        from focal_loss import BinaryFocalLoss
        print('✓ focal_loss already available')
        return True
    except ImportError:
        print('Installing focal_loss...')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'focal-loss', '--user'])
            print('✓ focal_loss installed successfully')
            return True
        except subprocess.CalledProcessError:
            print('Attempting alternative installation...')
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'focal-loss-tensorflow', '--user'])
                print('✓ focal-loss-tensorflow installed successfully')
                return True
            except subprocess.CalledProcessError:
                print('✗ Could not install focal_loss via pip')
                print('Creating custom implementation...')
                return False

if not install_focal_loss():
    # Create custom focal loss implementation
    with open('focal_loss.py', 'w') as f:
        f.write('''import tensorflow as tf
from tensorflow.keras import backend as K

class BinaryFocalLoss:
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Calculate focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)

        focal_loss = -alpha_t * K.pow(1 - pt, self.gamma) * K.log(pt)

        return K.mean(focal_loss)
''')
    print('✓ Created custom focal_loss.py')
"
echo "==============================="
echo ""

# =======================================================================
# EXECUTE TRAINING
# =======================================================================

echo "🚀 STARTING OPTIMIZED MITOCHONDRIA SEGMENTATION TRAINING"
echo "=============================================="
echo "Training 3 models sequentially with optimized hyperparameters:"
echo "1. UNet (LR=1e-3, BS=8, up to 100 epochs)"
echo "2. Attention UNet (LR=1e-4, BS=16, up to 100 epochs) - BEST"
echo "3. Attention ResUNet (LR=5e-4, BS=16, up to 100 epochs)"
echo ""
echo "Early stopping enabled (patience=15)"
echo "Expected total time: 6-12 hours (depending on convergence)"
echo "Results will be saved in timestamped directory"
echo "=============================================="
echo ""

# Execute the training with enhanced error handling and logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
singularity exec --nv "$image" python3 microscope_optimized_training.py 2>&1 | tee "microscope_training_${TIMESTAMP}.log"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "OPTIMIZED MICROSCOPE TRAINING COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✓ SUCCESS"
    echo ""
    echo "✓ Training completed successfully!"
    echo ""

    # Find the output directory (most recent microscope_training_*)
    OUTPUT_DIR=$(ls -td microscope_training_* 2>/dev/null | head -1)

    if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
        echo "Generated files in $OUTPUT_DIR:"
        echo ""
        echo "📁 Model files (.hdf5):"
        ls -lh "$OUTPUT_DIR"/*.hdf5 2>/dev/null || echo "   No model files found"
        echo ""
        echo "📊 Training history (.csv):"
        ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "   No history files found"
        echo ""
        echo "🖼️ Visualization files (.png):"
        ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null || echo "   No plot files found"
        echo ""

        echo "🎯 MODEL PERFORMANCE SUMMARY:"
        echo "=============================="
        singularity exec --nv "$image" python3 -c "
import pandas as pd
import os

output_dir = '$OUTPUT_DIR'
csv_files = {
    'UNet': os.path.join(output_dir, 'unet_history.csv'),
    'Attention_UNet': os.path.join(output_dir, 'attention_unet_history.csv'),
    'Attention_ResUNet': os.path.join(output_dir, 'attention_resunet_history.csv')
}

print('Model Performance on Microscope Dataset:')
print('-' * 70)

results = []
for model_name, csv_file in csv_files.items():
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if 'val_jacard_coef' in df.columns:
                best_val_jacard = df['val_jacard_coef'].max()
                best_epoch = df['val_jacard_coef'].idxmax() + 1
                final_loss = df['val_loss'].iloc[-1]
                total_epochs = len(df)
                results.append({
                    'Model': model_name,
                    'Best_Jaccard': best_val_jacard,
                    'Epoch': best_epoch,
                    'Total_Epochs': total_epochs,
                    'Final_Loss': final_loss
                })
        except Exception as e:
            print(f'Error reading {csv_file}: {e}')

if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Best_Jaccard', ascending=False)

    print(results_df.to_string(index=False))
    print()
    print(f\"🏆 Best Model: {results_df.iloc[0]['Model']}\")
    print(f\"   Best Val Jaccard: {results_df.iloc[0]['Best_Jaccard']:.4f}\")
    print(f\"   Achieved at Epoch: {results_df.iloc[0]['Epoch']:.0f}\")
else:
    print('No training results found.')
" 2>/dev/null || echo "Unable to generate performance summary"
    else
        echo "Output directory not found. Results may be in current directory."
        echo "Looking for generated files..."
        ls -lh microscope_training_*/ 2>/dev/null || echo "No output directories found"
    fi

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Training failed!"
    echo ""
    echo "🔍 ERROR ANALYSIS:"
    echo "=================="
    echo "Check the console log for detailed error information:"
    echo "   microscope_training_${TIMESTAMP}.log"
    echo ""

    if [ -f "microscope_training_${TIMESTAMP}.log" ]; then
        echo "Last 30 lines of console log:"
        tail -30 "microscope_training_${TIMESTAMP}.log"
    fi
    echo ""

    echo "🔧 COMMON ISSUES AND SOLUTIONS:"
    echo "==============================="
    echo "1. Dataset path issues: Ensure ./dataset_microscope/images/ and ./dataset_microscope/masks/ exist"
    echo "2. Missing image files: Check that your dataset contains supported image files (.tif, .png, .jpg)"
    echo "3. Memory issues: Models use different batch sizes (8 and 16)"
    echo "4. Dependency issues: focal_loss package installation may have failed"
    echo "5. GPU memory: Try reducing batch size if out of memory"
    echo "6. Image format: Ensure images are readable by cv2/PIL"
fi

echo ""
echo "📝 CONSOLE LOG: microscope_training_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "MICROSCOPE OPTIMIZED TRAINING COMPLETE"
echo "Dataset: dataset_microscope/"
echo "Models: UNet, Attention UNet, Attention ResUNet"
echo "Optimized Hyperparameters Applied"
echo "======================================="
