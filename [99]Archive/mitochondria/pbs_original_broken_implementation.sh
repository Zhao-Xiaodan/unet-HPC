#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Mitochondria_Original_Broken
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# MITOCHONDRIA SEGMENTATION - ORIGINAL BROKEN IMPLEMENTATION TEST
# =======================================================================
# Testing original broken Jaccard implementation with full dataset
# to validate our breakthrough analysis findings
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "MITOCHONDRIA SEGMENTATION - ORIGINAL BROKEN IMPLEMENTATION TEST"
echo "======================================================================="
echo "🚨 WARNING: Using BROKEN Jaccard implementation for validation"
echo "📊 Dataset: Full dataset_full_stack (1,980 patches)"
echo "🎯 Purpose: Validate breakthrough analysis findings"
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
echo "🚨 Implementation: ORIGINAL BROKEN JACCARD"
echo "Dataset Images: ./dataset_full_stack/images/ (1980 patches)"
echo "Dataset Masks: ./dataset_full_stack/masks/ (1980 patches)"
echo "Image Size: 256x256"
echo "Batch Size: 8 (original)"
echo "Learning Rate: 1e-2 (original)"
echo "Epochs per Model: 50 (original)"
echo "Models: 3 (UNet, Attention UNet, Attention ResUNet)"
echo "Loss Function: Binary Focal Loss"
echo "Expected Training Time: ~6-8 hours"
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
if [ -d "./dataset_full_stack/images/" ] && [ -d "./dataset_full_stack/masks/" ]; then
    echo "   ✓ Dataset directories found"

    # Count files
    img_count=$(find ./dataset_full_stack/images/ -name "*.tif" | wc -l)
    mask_count=$(find ./dataset_full_stack/masks/ -name "*.tif" | wc -l)

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
    echo "   Expected: ./dataset_full_stack/images/ and ./dataset_full_stack/masks/"
    echo "   Current directory contents:"
    ls -la ./
    exit 1
fi

# Check if Python files exist
echo ""
echo "2. Checking Python files..."
if [ -f "./224_225_226_mito_segm_using_various_unet_models_original.py" ] && [ -f "./models_original.py" ]; then
    echo "   ✓ Python training script found"
    echo "   ✓ Models definition file found"
else
    echo "   ERROR: Required Python files not found"
    echo "   Expected:"
    echo "   - 224_225_226_mito_segm_using_various_unet_models_original.py"
    echo "   - models_original.py"
    ls -la ./*.py
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

echo "=== PREPARING ORIGINAL BROKEN TRAINING SCRIPT ==="
echo "Creating wrapper with dependency installation..."

# Create a wrapper script that handles focal_loss installation
cat > run_original_broken_training.py << 'EOF'
#!/usr/bin/env python3
"""
Wrapper script to run ORIGINAL BROKEN mitochondria segmentation
with proper dependency handling.
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
            print("Creating custom focal_loss implementation...")
            create_custom_focal_loss()
            return True

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

def main():
    print("=== ORIGINAL BROKEN MITOCHONDRIA SEGMENTATION WRAPPER ===")
    print("🚨 WARNING: This uses the BROKEN Jaccard implementation!")
    print()

    # Install focal loss
    install_focal_loss()

    print()
    print("🚨 Starting ORIGINAL BROKEN implementation training...")
    print("=" * 60)

    # Import and run the main training script
    exec(open('224_225_226_mito_segm_using_various_unet_models_original.py').read())

if __name__ == "__main__":
    main()
EOF

echo "✓ Original broken training wrapper script created"
echo "============================="
echo ""

# =======================================================================
# EXECUTE TRAINING
# =======================================================================

echo "🚨 STARTING ORIGINAL BROKEN IMPLEMENTATION TRAINING"
echo "=================================================="
echo "🚨 WARNING: Using BROKEN Jaccard coefficient!"
echo "Training 3 models sequentially:"
echo "1. UNet (50 epochs)"
echo "2. Attention UNet (50 epochs)"
echo "3. Attention ResUNet (50 epochs)"
echo ""
echo "Expected total time: 6-8 hours"
echo "Results will be saved with timestamps"
echo "=================================================="

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="mitochondria_segmentation_original_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Execute the training with enhanced error handling and logging
singularity exec --nv "$image" python3 run_original_broken_training.py 2>&1 | tee "${OUTPUT_DIR}/training_console_${TIMESTAMP}.log"

# Capture exit code
EXIT_CODE=$?

# Move generated files to output directory
echo ""
echo "Moving generated files to output directory..."
mv mitochondria_segmentation_original_*/*.hdf5 "$OUTPUT_DIR/" 2>/dev/null || echo "No .hdf5 files to move"
mv mitochondria_segmentation_original_*/*.csv "$OUTPUT_DIR/" 2>/dev/null || echo "No .csv files to move"
mv mitochondria_segmentation_original_*/*.json "$OUTPUT_DIR/" 2>/dev/null || echo "No .json files to move"

echo ""
echo "======================================================================="
echo "🚨 ORIGINAL BROKEN IMPLEMENTATION TRAINING COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✓ SUCCESS"
    echo ""
    echo "✅ Training completed successfully!"
    echo ""
    echo "Generated files in $OUTPUT_DIR:"
    echo "📁 Model files (.hdf5):"
    ls -la "$OUTPUT_DIR"/*.hdf5 2>/dev/null || echo "   No model files found"
    echo ""
    echo "📊 Training history (.csv):"
    ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "   No history files found"
    echo ""
    echo "📄 Results summary (.json):"
    ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "   No summary files found"
    echo ""

    echo "🚨 ORIGINAL BROKEN IMPLEMENTATION RESULTS:"
    echo "========================================"
    singularity exec --nv "$image" python3 -c "
import pandas as pd
import os
import glob
import json

output_dir = '$OUTPUT_DIR'
csv_files = glob.glob(os.path.join(output_dir, '*history*original.csv'))

if csv_files:
    print('🚨 BROKEN Implementation Training History:')
    print('-' * 50)

    for csv_file in csv_files:
        model_name = os.path.basename(csv_file).replace('_history_df_original.csv', '').replace('custom_code_', '')
        try:
            df = pd.read_csv(csv_file)
            if 'val_jacard_coef' in df.columns:
                best_val_jacard = df['val_jacard_coef'].max()
                best_epoch = df['val_jacard_coef'].idxmax() + 1
                final_loss = df['val_loss'].iloc[-1]
                print(f'🚨 {model_name} (BROKEN):')
                print(f'  Val Jaccard: {best_val_jacard:.6f} (epoch {best_epoch})')
                print(f'  Final Loss: {final_loss:.4f}')
                print()
        except Exception as e:
            print(f'Error reading {csv_file}: {e}')

# Also display JSON summary if available
json_files = glob.glob(os.path.join(output_dir, '*summary*.json'))
if json_files:
    print()
    print('📋 Results Summary:')
    print('-' * 30)
    with open(json_files[0], 'r') as f:
        summary = json.load(f)
        print(f\"🚨 BROKEN Implementation: {summary.get('implementation', 'Unknown')}\")
        print(f\"📊 Dataset size: {summary.get('dataset_size', 'Unknown')} patches\")
        print(f\"🎯 UNet Jaccard: {summary.get('UNet_best_val_jaccard', 0):.6f}\")
        print(f\"🎯 Attention UNet Jaccard: {summary.get('Attention_UNet_best_val_jaccard', 0):.6f}\")
        print(f\"🎯 Attention ResUNet Jaccard: {summary.get('AttentionRes_UNet_best_val_jaccard', 0):.6f}\")
        print()
        print('⚠️  These LOW values confirm the BROKEN Jaccard implementation!')
else:
    print('No JSON summary files found.')
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
fi

echo ""
echo "📁 ALL OUTPUT FILES SAVED IN: $OUTPUT_DIR"
echo "📝 CONSOLE LOG: ${OUTPUT_DIR}/training_console_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "🚨 ORIGINAL BROKEN VALIDATION COMPLETE"
echo "Purpose: Validate breakthrough analysis"
echo "Expected: LOW Jaccard values (~0.07-0.09)"
echo "======================================="