#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Xukuang_Shrunk
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# TRAIN SHRUNK MASKS DATASET - XUKUANG'S PARAMETERS
# =======================================================================
# Trains 3 U-Net architectures using Xukuang's parameters from bead_seg.ipynb
#
# Models:
# - Standard U-Net
# - Attention U-Net
# - Attention Residual U-Net
#
# Training Parameters (from bead_seg.ipynb):
# - Learning Rate: 5e-3
# - Epochs: 200
# - Batch Size: 4
# - Image Size: 512×512
# - Loss: BinaryFocalLoss(gamma=2)
# - Optimizer: Adam
# =======================================================================

cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "TRAIN SHRUNK MASKS - XUKUANG'S PARAMETERS"
echo "======================================================================="
echo "Script: train_shrunk_xukuang_parameters.py"
echo "PBS Script: pbs_train_shrunk_xukuang_parameters.sh"
echo ""
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $(free -h | grep Mem | awk '{print $2}'), CPUs: $(nproc)"
echo ""

# =======================================================================
# TRAINING CONFIGURATION
# =======================================================================

echo "=== TRAINING CONFIGURATION ==="
echo "Dataset: dataset_shrunk_masks/"
echo "Source: bead_seg.ipynb (Xukuang parameters)"
echo ""
echo "Architectures:"
echo "  1. U-Net"
echo "  2. Attention U-Net"
echo "  3. Attention Residual U-Net"
echo ""
echo "Parameters:"
echo "  Learning Rate:  5e-3"
echo "  Epochs:         200"
echo "  Batch Size:     4"
echo "  Image Size:     512×512"
echo "  Loss Function:  BinaryFocalLoss(gamma=2)"
echo "  Optimizer:      Adam"
echo "  Test Split:     20%"
echo "==============================="
echo ""

# =======================================================================
# ENVIRONMENT SETUP
# =======================================================================

echo "=== ENVIRONMENT SETUP ==="

export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export CUDA_VISIBLE_DEVICES=0

module load singularity

image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

if [ ! -f "$image" ]; then
    echo "ERROR: TensorFlow container not found at $image"
    exit 1
fi

echo "✓ TensorFlow Container: $image"
echo "==========================="
echo ""

# =======================================================================
# PRE-EXECUTION CHECKS
# =======================================================================

echo "=== PRE-EXECUTION CHECKS ==="

echo "1. Checking Python scripts..."
if [ ! -f "./train_shrunk_xukuang_parameters.py" ]; then
    echo "   ✗ ERROR: train_shrunk_xukuang_parameters.py not found!"
    exit 1
fi
echo "   ✓ train_shrunk_xukuang_parameters.py found"

if [ ! -f "./models.py" ]; then
    echo "   ✗ ERROR: models.py not found!"
    exit 1
fi
echo "   ✓ models.py found"

if [ ! -f "./focal_loss.py" ]; then
    echo "   ✗ ERROR: focal_loss.py not found!"
    exit 1
fi
echo "   ✓ focal_loss.py found"

echo ""
echo "2. Checking dataset..."
if [ ! -d "./dataset_shrunk_masks/" ]; then
    echo "   ✗ ERROR: dataset_shrunk_masks directory not found!"
    exit 1
fi
echo "   ✓ dataset_shrunk_masks directory found"

if [ ! -d "./dataset_shrunk_masks/images/" ]; then
    echo "   ✗ ERROR: dataset_shrunk_masks/images/ not found!"
    exit 1
fi

if [ ! -d "./dataset_shrunk_masks/masks/" ]; then
    echo "   ✗ ERROR: dataset_shrunk_masks/masks/ not found!"
    exit 1
fi

image_count=$(find ./dataset_shrunk_masks/images/ -name "*.png" -o -name "*.jpg" -o -name "*.tif" 2>/dev/null | wc -l)
mask_count=$(find ./dataset_shrunk_masks/masks/ -name "*.png" -o -name "*.jpg" -o -name "*.tif" 2>/dev/null | wc -l)

echo "   ✓ Found $image_count images"
echo "   ✓ Found $mask_count masks"

if [ $image_count -eq 0 ]; then
    echo "   ✗ ERROR: No images found in dataset_shrunk_masks/images/"
    exit 1
fi

if [ $mask_count -eq 0 ]; then
    echo "   ✗ ERROR: No masks found in dataset_shrunk_masks/masks/"
    exit 1
fi

if [ $image_count -ne $mask_count ]; then
    echo "   ⚠ WARNING: Image count ($image_count) != Mask count ($mask_count)"
fi

echo "=============================="
echo ""

# =======================================================================
# TENSORFLOW & GPU CHECK
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

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('✓ GPU memory growth enabled')
    except RuntimeError as e:
        print('Memory growth setting:', e)

    # Test GPU operation
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[2.0, 0.0], [0.0, 2.0]])
            c = tf.matmul(a, b)
        print('✓ Basic GPU operation successful')
    except Exception as e:
        print('✗ GPU operation failed:', e)
        sys.exit(1)
else:
    print('⚠ WARNING: No GPU detected!')

# Check dependencies
print()
print('Checking dependencies:')
deps = ['cv2', 'PIL', 'matplotlib', 'numpy', 'pandas', 'sklearn']
for dep in deps:
    try:
        __import__(dep)
        print(f'  ✓ {dep}')
    except ImportError:
        print(f'  ✗ {dep} - MISSING!')
        sys.exit(1)

# Check focal_loss
try:
    from focal_loss import BinaryFocalLoss
    print('  ✓ focal_loss.BinaryFocalLoss')
except ImportError:
    print('  ✗ focal_loss.BinaryFocalLoss - MISSING!')
    sys.exit(1)

print()
print('✓ All dependencies satisfied')
"

CHECK_EXIT=$?
if [ $CHECK_EXIT -ne 0 ]; then
    echo ""
    echo "✗ Environment check failed!"
    exit 1
fi

echo "================================"
echo ""

# =======================================================================
# EXECUTE TRAINING
# =======================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🚀 STARTING TRAINING"
echo "=============================================="
echo "Phase 1: Load dataset (98 images + 98 masks)"
echo "Phase 2: Train-test split (80/20)"
echo "Phase 3: Train U-Net (200 epochs)"
echo "Phase 4: Train Attention U-Net (200 epochs)"
echo "Phase 5: Train Attention ResUNet (200 epochs)"
echo ""
echo "Expected duration: ~12-18 hours (depends on GPU)"
echo ""
echo "Progress logged below..."
echo "=============================================="
echo ""

singularity exec --nv "$image" python3 train_shrunk_xukuang_parameters.py 2>&1 | tee "train_shrunk_console_${TIMESTAMP}.log"

EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "TRAINING COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✓ SUCCESS"
    echo ""
    echo "✓ Training completed successfully!"
    echo ""

    # Find output directory
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "xukuang_params_shrunk_*" -mtime -1 | sort | tail -1)

    if [ -d "$OUTPUT_DIR" ]; then
        echo "📁 RESULTS DIRECTORY: $OUTPUT_DIR"
        echo ""

        # Copy logs and scripts to output directory
        echo "📋 Archiving logs and scripts..."
        if [ -n "$PBS_JOBID" ]; then
            PBS_OUTPUT_FILE="${PBS_JOBNAME}.o${PBS_JOBID}"
            if [ -f "$PBS_OUTPUT_FILE" ]; then
                cp "$PBS_OUTPUT_FILE" "$OUTPUT_DIR/"
                echo "   ✓ Copied PBS output: $PBS_OUTPUT_FILE"
            fi
        fi

        if [ -f "train_shrunk_console_${TIMESTAMP}.log" ]; then
            cp "train_shrunk_console_${TIMESTAMP}.log" "$OUTPUT_DIR/"
            echo "   ✓ Copied console log"
        fi

        cp train_shrunk_xukuang_parameters.py "$OUTPUT_DIR/"
        cp pbs_train_shrunk_xukuang_parameters.sh "$OUTPUT_DIR/"
        echo "   ✓ Copied source scripts"
        echo ""

        # Check results
        echo "📊 GENERATED FILES:"

        echo "   Models:"
        ls "$OUTPUT_DIR"/*.keras 2>/dev/null | sed 's/^/     - /'

        echo "   Training History:"
        ls "$OUTPUT_DIR"/*_history.csv 2>/dev/null | sed 's/^/     - /'

        echo "   Metadata:"
        ls "$OUTPUT_DIR"/*.json 2>/dev/null | sed 's/^/     - /'

        echo ""
        echo "🎯 NEXT STEPS:"
        echo "============="
        echo "1. View training summary:"
        echo "   cat $OUTPUT_DIR/TRAINING_SUMMARY.json"
        echo ""
        echo "2. Plot training curves:"
        echo "   python plot_training_history.py $OUTPUT_DIR"
        echo ""
        echo "3. Evaluate models on test set:"
        echo "   python evaluate_models.py $OUTPUT_DIR"
        echo ""
        echo "4. Compare with original hyperparameter search results"

    else
        echo "⚠ WARNING: Expected output directory not found!"
        echo "   Looking for: xukuang_params_shrunk_*"
    fi

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Training failed!"
    echo ""

    if [ -f "train_shrunk_console_${TIMESTAMP}.log" ]; then
        echo "Last 50 lines of console log:"
        echo "-----------------------------"
        tail -50 "train_shrunk_console_${TIMESTAMP}.log"
    fi

    echo ""
    echo "🔧 COMMON ISSUES:"
    echo "================="
    echo "1. Dataset loading errors:"
    echo "   - Check that images and masks exist in dataset_shrunk_masks/"
    echo "   - Verify image formats (.png, .jpg, .tif)"
    echo "   - Check that images and masks have matching filenames"
    echo ""
    echo "2. GPU memory issues:"
    echo "   - Check nvidia-smi for available memory"
    echo "   - Reduce batch size from 4 to 2 if OOM errors occur"
    echo ""
    echo "3. Missing dependencies:"
    echo "   - Verify focal_loss.py exists"
    echo "   - Verify models.py exists"
    echo "   - Check all required Python packages are installed"
fi

echo ""
echo "📝 CONSOLE LOG: train_shrunk_console_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "TRAINING JOB COMPLETE"
echo "Xukuang Parameters + Shrunk Masks"
echo "======================================="
