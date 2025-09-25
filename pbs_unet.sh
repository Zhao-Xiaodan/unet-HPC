#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Mitochondria_UNet_Segmentation
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# MITOCHONDRIA SEGMENTATION USING VARIOUS UNET MODELS - PBS SCRIPT
# =======================================================================
# Training UNet, Attention UNet, and Attention ResUNet for mitochondria segmentation
# Based on the provided Python scripts
# =======================================================================

echo "======================================================================="
echo "MITOCHONDRIA SEGMENTATION - UNET MODELS TRAINING"
echo "======================================================================="
echo "Models: UNet, Attention UNet, Attention ResUNet"
echo "Task: Mitochondria semantic segmentation"
echo "Framework: TensorFlow/Keras"
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
echo "Dataset Images: ./dataset/images/"
echo "Dataset Masks: ./dataset/masks/"
echo "Image Size: 256x256"
echo "Batch Size: 8"
echo "Epochs per Model: 50"
echo "Models: 3 (UNet, Attention UNet, Attention ResUNet)"
echo "Loss Function: Binary Focal Loss"
echo "Optimizer: Adam (lr=1e-2)"
echo "Expected Training Time: ~8-12 hours"
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
if [ -d "./dataset/images/" ] && [ -d "./dataset/masks/" ]; then
    echo "   ✓ Dataset directories found"

    # Count files
    img_count=$(find ./dataset/images/ -name "*.tif" | wc -l)
    mask_count=$(find ./dataset/masks/ -name "*.tif" | wc -l)

    echo "   ✓ Images found: $img_count .tif files"
    echo "   ✓ Masks found: $mask_count .tif files"

    if [ $img_count -eq 0 ] || [ $mask_count -eq 0 ]; then
        echo "   WARNING: No .tif files found in dataset directories"
        echo "   Please ensure your dataset contains .tif files as expected by the code"
    fi

    if [ $img_count -ne $mask_count ]; then
        echo "   WARNING: Unequal number of images ($img_count) and masks ($mask_count)"
        echo "   This may cause issues during training"
    fi
else
    echo "   ERROR: Dataset directories not found!"
    echo "   Expected: ./dataset/images/ and ./dataset/masks/"
    echo "   Current directory contents:"
    ls -la ./
    exit 1
fi

# Check if Python files exist
echo ""
echo "2. Checking Python files..."
if [ -f "./224_225_226_mito_segm_using_various_unet_models.py" ] && [ -f "./224_225_226_models.py" ]; then
    echo "   ✓ Python training script found"
    echo "   ✓ Models definition file found"
else
    echo "   ERROR: Required Python files not found"
    echo "   Expected:"
    echo "   - 224_225_226_mito_segm_using_various_unet_models.py"
    echo "   - 224_225_226_models.py (should be renamed to models.py)"
    ls -la ./*.py
    exit 1
fi

# Rename models file if needed
if [ -f "./224_225_226_models.py" ] && [ ! -f "./models.py" ]; then
    echo "   ✓ Renaming 224_225_226_models.py to models.py"
    cp ./224_225_226_models.py ./models.py
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
print('GPU available:', tf.test.is_gpu_available())

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
# CREATE MODIFIED TRAINING SCRIPT
# =======================================================================

echo "=== PREPARING TRAINING SCRIPT ==="
echo "Creating dataset path fix and dependency installation..."

# Create a wrapper script that fixes the dataset path issue
cat > run_mitochondria_training.py << 'EOF'
#!/usr/bin/env python3
"""
Wrapper script to run mitochondria segmentation with proper dataset paths
and dependency handling.
"""

import os
import sys
import subprocess

def install_focal_loss():
    """Install focal_loss if not available"""
    try:
        from focal_loss import BinaryFocalLoss
        print("✓ focal_loss already available")
        return True
    except ImportError:
        print("Installing focal_loss...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "focal-loss"])
            print("✓ focal_loss installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install focal_loss: {e}")
            print("Attempting alternative installation...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "focal-loss-tensorflow"])
                print("✓ focal-loss-tensorflow installed successfully")
                return True
            except subprocess.CalledProcessError:
                print("✗ Could not install focal_loss. Will implement custom version.")
                return False

def create_custom_focal_loss():
    """Create custom focal loss implementation if package not available"""
    with open('focal_loss.py', 'w') as f:
        f.write('''
import tensorflow as tf
from tensorflow.keras import backend as K

class BinaryFocalLoss:
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        """
        Binary focal loss implementation
        """
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Calculate focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)

        focal_loss = -alpha_t * K.pow(1 - pt, self.gamma) * K.log(pt)

        return K.mean(focal_loss)
''')

def fix_dataset_paths():
    """Create symbolic links to fix dataset path mismatch"""
    if not os.path.exists('data'):
        if os.path.exists('dataset'):
            print("Creating symbolic link: data -> dataset")
            os.symlink('dataset', 'data')
        else:
            print("ERROR: Neither 'data' nor 'dataset' directory found!")
            return False
    return True

def main():
    print("=== MITOCHONDRIA SEGMENTATION TRAINING WRAPPER ===")
    print()

    # Fix dataset paths
    if not fix_dataset_paths():
        sys.exit(1)

    # Install or create focal loss
    if not install_focal_loss():
        create_custom_focal_loss()
        print("✓ Created custom focal_loss implementation")

    print()
    print("Starting mitochondria segmentation training...")
    print("=" * 50)

    # Import and run the main training script
    exec(open('224_225_226_mito_segm_using_various_unet_models.py').read())

if __name__ == "__main__":
    main()
EOF

echo "✓ Training wrapper script created"
echo "============================="
echo ""

# =======================================================================
# EXECUTE TRAINING
# =======================================================================

echo "🚀 STARTING MITOCHONDRIA SEGMENTATION TRAINING"
echo "=============================================="
echo "Training 3 models sequentially:"
echo "1. UNet (50 epochs)"
echo "2. Attention UNet (50 epochs)"
echo "3. Attention ResUNet (50 epochs)"
echo ""
echo "Expected total time: 8-12 hours"
echo "Results will be saved as .hdf5 model files"
echo "=============================================="

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="mitochondria_segmentation_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Change to working directory
cd /home/svu/phyzxi/scratch/densityCNN-HPC

# Execute the training with enhanced error handling and logging
singularity exec --nv "$image" python3 run_mitochondria_training.py 2>&1 | tee "${OUTPUT_DIR}/training_console_${TIMESTAMP}.log"

# Capture exit code
EXIT_CODE=$?

# Move generated files to output directory
echo ""
echo "Moving generated files to output directory..."
mv *.hdf5 "$OUTPUT_DIR/" 2>/dev/null || echo "No .hdf5 files to move"
mv *.csv "$OUTPUT_DIR/" 2>/dev/null || echo "No .csv files to move"
mv *.png "$OUTPUT_DIR/" 2>/dev/null || echo "No .png files to move"

echo ""
echo "======================================================================="
echo "MITOCHONDRIA SEGMENTATION TRAINING COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✓ SUCCESS"
    echo ""
    echo "✓ Training completed successfully!"
    echo ""
    echo "Generated files in $OUTPUT_DIR:"
    echo "📁 Model files (.hdf5):"
    ls -la "$OUTPUT_DIR"/*.hdf5 2>/dev/null || echo "   No model files found"
    echo ""
    echo "📊 Training history (.csv):"
    ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "   No history files found"
    echo ""
    echo "🖼️ Visualization files (.png):"
    ls -la "$OUTPUT_DIR"/*.png 2>/dev/null || echo "   No plot files found"
    echo ""

    echo "🎯 MODEL PERFORMANCE SUMMARY:"
    echo "=============================="
    singularity exec --nv "$image" python3 -c "
import pandas as pd
import os
import glob

output_dir = '$OUTPUT_DIR'
csv_files = glob.glob(os.path.join(output_dir, '*history*.csv'))

if csv_files:
    print('Training History Summary:')
    print('-' * 40)

    for csv_file in csv_files:
        model_name = os.path.basename(csv_file).replace('_history_df.csv', '').replace('custom_code_', '')
        try:
            df = pd.read_csv(csv_file)
            if 'val_jacard_coef' in df.columns:
                best_val_jacard = df['val_jacard_coef'].max()
                best_epoch = df['val_jacard_coef'].idxmax() + 1
                final_loss = df['val_loss'].iloc[-1]
                print(f'{model_name}:')
                print(f'  Best Val Jaccard: {best_val_jacard:.4f} (epoch {best_epoch})')
                print(f'  Final Val Loss: {final_loss:.4f}')
                print()
        except Exception as e:
            print(f'Error reading {csv_file}: {e}')
else:
    print('No training history CSV files found.')
" 2>/dev/null || echo "Unable to generate performance summary"

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Training failed!"
    echo ""
    echo "🔍 ERROR ANALYSIS:"
    echo "=================="
    echo "Check the console log for detailed error information:"
    echo "   ${OUTPUT_DIR}/training_console_${TIMESTAMP}.log"
    echo ""

    if [ -f "${OUTPUT_DIR}/training_console_${TIMESTAMP}.log" ]; then
        echo "Last 20 lines of console log:"
        tail -20 "${OUTPUT_DIR}/training_console_${TIMESTAMP}.log"
    fi
    echo ""

    echo "🔧 COMMON ISSUES AND SOLUTIONS:"
    echo "==============================="
    echo "1. Dataset path issues: Ensure ./dataset/images/ and ./dataset/masks/ exist"
    echo "2. Missing .tif files: Check that your dataset contains .tif image files"
    echo "3. Memory issues: Reduce batch_size in the Python script"
    echo "4. Dependency issues: focal_loss package may need manual installation"
    echo "5. GPU memory: Try reducing image size or batch size"
fi

echo ""
echo "📁 ALL OUTPUT FILES SAVED IN: $OUTPUT_DIR"
echo "📝 CONSOLE LOG: ${OUTPUT_DIR}/training_console_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "MITOCHONDRIA SEGMENTATION JOB COMPLETE"
echo "Models: UNet, Attention UNet, Attention ResUNet"
echo "Framework: TensorFlow/Keras"
echo "======================================="
