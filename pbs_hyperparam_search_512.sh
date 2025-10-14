#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -N HyperSearch_512
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# HYPERPARAMETER SEARCH FOR 512×512 IMAGES WITH OOM PROTECTION
# =======================================================================
# Dataset: dataset_shrunk_masks (98 images, 512×512 RGB)
# Memory: Mixed precision + reduced filters + small batches
# OOM handling: Automatic batch size reduction on failure
# =======================================================================

cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "HYPERPARAMETER SEARCH - 512×512 IMAGES WITH OOM PROTECTION"
echo "======================================================================="
echo "Dataset: dataset_shrunk_masks (98 images)"
echo "Image Size: 512×512 (4× larger than previous 256×256!)"
echo "Framework: TensorFlow/Keras with Mixed Precision (FP16)"
echo ""

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $(free -h | grep Mem | awk '{print $2}'), CPUs: $(nproc)"
echo ""

# =======================================================================
# MEMORY MANAGEMENT STRATEGY
# =======================================================================

echo "=== MEMORY MANAGEMENT STRATEGY ==="
echo "Challenge: 512×512 images use 4× more memory than 256×256"
echo ""
echo "Solutions implemented:"
echo "  1. Mixed Precision Training (FP16)"
echo "     → Reduces memory by ~40%"
echo "     → Speeds up training by ~2×"
echo ""
echo "  2. Reduced Filter Count"
echo "     → Filters: 32 (vs 64 in 256×256)"
echo "     → Memory reduction: ~4×"
echo ""
echo "  3. Small Batch Sizes"
echo "     → Batch sizes: 2, 4 (vs 4, 8, 16 in 256×256)"
echo "     → With gradient accumulation to simulate larger batches"
echo ""
echo "  4. OOM Auto-Recovery"
echo "     → Catches ResourceExhaustedError"
echo "     → Reduces batch size from 4→2→1"
echo "     → Retries up to 2 times per configuration"
echo ""
echo "  5. GPU Memory Growth"
echo "     → Allocates memory as needed"
echo "     → Prevents preallocation of all GPU memory"
echo "=================================="
echo ""

# =======================================================================
# SEARCH SPACE (CONSERVATIVE FOR 512×512)
# =======================================================================

echo "=== HYPERPARAMETER SEARCH SPACE ==="
echo "Architectures: U-Net, ResUNet, Attention ResUNet"
echo ""
echo "Learning Rates: [1e-4, 5e-5]"
echo "  → Higher LR for faster convergence with large images"
echo ""
echo "Dropouts: [0.2, 0.3]"
echo "  → Conservative range (based on 256×256 findings)"
echo ""
echo "Batch Sizes: [2, 4]"
echo "  → SMALL batches required for 512×512!"
echo "  → Will auto-reduce to 1 if OOM occurs"
echo ""
echo "Total Configurations: 3 arch × 2 LR × 2 dropout × 2 BS = 24"
echo "Total Training Runs: 24 configs × 3 folds = 72 models"
echo ""
echo "Expected Runtime: 12-18 hours"
echo "  (Longer than 256×256 due to 4× larger images)"
echo "===================================="
echo ""

# =======================================================================
# ENVIRONMENT SETUP
# =======================================================================

echo "=== ENVIRONMENT SETUP ==="

export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

module load singularity

image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

if [ ! -f "$image" ]; then
    echo "ERROR: TensorFlow container not found at $image"
    exit 1
fi

echo "✓ TensorFlow Container: $image"
echo "✓ GPU memory growth: ENABLED"
echo "✓ Mixed precision: Will be enabled in Python script"
echo "==========================="
echo ""

# =======================================================================
# PRE-EXECUTION CHECKS
# =======================================================================

echo "=== PRE-EXECUTION CHECKS ==="

echo "1. Checking Python scripts..."
if [ ! -f "./hyperparameter_search_512.py" ]; then
    echo "   ✗ ERROR: hyperparameter_search_512.py not found!"
    exit 1
fi
echo "   ✓ hyperparameter_search_512.py found"

if [ ! -f "./model_architectures.py" ]; then
    echo "   ✗ ERROR: model_architectures.py not found!"
    exit 1
fi
echo "   ✓ model_architectures.py found"

if [ ! -f "./loss_functions_fixed.py" ]; then
    echo "   ✗ ERROR: loss_functions_fixed.py not found!"
    exit 1
fi
echo "   ✓ loss_functions_fixed.py found"

echo ""
echo "2. Checking dataset..."
if [ ! -d "./dataset_shrunk_masks/images/" ] || [ ! -d "./dataset_shrunk_masks/masks/" ]; then
    echo "   ✗ ERROR: dataset_shrunk_masks not found!"
    exit 1
fi

img_count=$(find ./dataset_shrunk_masks/images/ -name "*.png" 2>/dev/null | wc -l)
mask_count=$(find ./dataset_shrunk_masks/masks/ -name "*.png" 2>/dev/null | wc -l)

echo "   ✓ Images: $img_count .png files"
echo "   ✓ Masks: $mask_count .png files"

if [ $img_count -eq 0 ] || [ $mask_count -eq 0 ]; then
    echo "   ✗ ERROR: No images found in dataset"
    exit 1
fi

# Check image size
echo ""
echo "3. Verifying image size..."
singularity exec --nv "$image" python3 -c "
from PIL import Image
import glob
images = glob.glob('./dataset_shrunk_masks/images/*.png')
if images:
    img = Image.open(images[0])
    print(f'   ✓ Sample image size: {img.size}')
    print(f'   ✓ Mode: {img.mode}')
    if img.size != (512, 512):
        print(f'   ⚠ WARNING: Expected 512×512, got {img.size}')
else:
    print('   ✗ No images found')
"

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

# Enable memory growth
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
    print('⚠ WARNING: No GPU detected! Training will be VERY slow.')

# Check dependencies
print()
print('Checking dependencies:')
deps = ['cv2', 'PIL', 'matplotlib', 'numpy', 'sklearn', 'pandas', 'tqdm']
for dep in deps:
    try:
        __import__(dep)
        print(f'  ✓ {dep}')
    except ImportError:
        print(f'  ✗ {dep} - MISSING!')
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
# EXECUTE HYPERPARAMETER SEARCH
# =======================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🚀 STARTING HYPERPARAMETER SEARCH"
echo "=============================================="
echo "Phase 1: Train 24 configurations × 3 folds = 72 models"
echo "Phase 2: Identify best configuration"
echo "Phase 3: Save results and summary"
echo ""
echo "Memory Safety Features:"
echo "  - Mixed precision (FP16) active"
echo "  - OOM auto-recovery enabled"
echo "  - Batch size will reduce if OOM detected"
echo ""
echo "Progress logged below..."
echo "=============================================="
echo ""

singularity exec --nv "$image" python3 hyperparameter_search_512.py 2>&1 | tee "hyperparam_search_512_console_${TIMESTAMP}.log"

EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "HYPERPARAMETER SEARCH COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✓ SUCCESS"
    echo ""
    echo "✓ Hyperparameter search completed successfully!"
    echo ""

    # Find output directory
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "hyperparameter_search_512_*" -mtime -1 | sort | tail -1)

    if [ -d "$OUTPUT_DIR" ]; then
        echo "📁 RESULTS DIRECTORY: $OUTPUT_DIR"
        echo ""

        # Check results
        if [ -f "$OUTPUT_DIR/summary.json" ]; then
            echo "📊 BEST CONFIGURATION FOUND:"
            singularity exec --nv "$image" python3 -c "
import json
try:
    with open('$OUTPUT_DIR/summary.json', 'r') as f:
        summary = json.load(f)
    print(f\"   Config: {summary['best_config']}\")
    print(f\"   Jaccard: {summary['best_jaccard']:.4f}\")
    print(f\"   Successful configs: {summary['successful_configs']}/{summary['total_configs']}\")
except Exception as e:
    print(f'Unable to parse summary: {e}')
" 2>/dev/null || echo "   (Summary file found)"
            echo ""
        fi

        # Count results
        if [ -f "$OUTPUT_DIR/all_results.csv" ]; then
            echo "📄 RESULTS CSV:"
            echo "   $OUTPUT_DIR/all_results.csv"
            echo ""
        fi

        echo "🎯 NEXT STEPS:"
        echo "============="
        echo "1. Review results:"
        echo "   cat $OUTPUT_DIR/summary.json"
        echo ""
        echo "2. Run density analysis with best model:"
        echo "   qsub pbs_density_analysis_512.sh"
        echo "   (Make sure to update the script with: $OUTPUT_DIR)"
        echo ""
        echo "3. If many OOM failures occurred:"
        echo "   - Check all_results.csv for 'success'=False entries"
        echo "   - Consider reducing filters further (32→24 or 16)"
        echo "   - Or reduce image size (512→384)"

    else
        echo "⚠ WARNING: Expected output directory not found!"
    fi

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Hyperparameter search failed!"
    echo ""

    if [ -f "hyperparam_search_512_console_${TIMESTAMP}.log" ]; then
        echo "Last 50 lines of console log:"
        echo "-----------------------------"
        tail -50 "hyperparam_search_512_console_${TIMESTAMP}.log"
    fi

    echo ""
    echo "🔧 COMMON ISSUES:"
    echo "================="
    echo "1. Out of Memory (OOM):"
    echo "   - Script should auto-reduce batch size"
    echo "   - If all configs fail, reduce filters in hyperparameter_search_512.py"
    echo "   - Change: 'filters': 32 → 'filters': 24 or 16"
    echo ""
    echo "2. Dataset issues:"
    echo "   - Check dataset_shrunk_masks/images/ and masks/"
    echo "   - Verify images are 512×512 PNG files"
    echo ""
    echo "3. GPU memory issues:"
    echo "   - Clear GPU: nvidia-smi (check if other jobs running)"
fi

echo ""
echo "📝 CONSOLE LOG: hyperparam_search_512_console_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "HYPERPARAMETER SEARCH JOB COMPLETE"
echo "Image Size: 512×512 with OOM Protection"
echo "======================================="
