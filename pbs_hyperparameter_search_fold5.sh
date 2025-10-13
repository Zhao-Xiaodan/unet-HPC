#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Hyperparam_Search
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# HYPERPARAMETER SEARCH FOR RESIDUAL ARCHITECTURES - PBS SCRIPT
# =======================================================================
# Searching optimal hyperparameters for ResUNet and Attention ResUNet
# Goal: Match or exceed U-Net's 69.94% Jaccard performance
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "HYPERPARAMETER SEARCH - RESIDUAL ARCHITECTURES"
echo "======================================================================="
echo "Target: ResUNet & Attention ResUNet"
echo "Goal: Find hyperparameters to match U-Net (69.94% Jaccard)"
echo "Method: Grid search over learning rate, dropout, batch size"
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
# CONFIGURATION
# =======================================================================

echo "=== SEARCH CONFIGURATION ==="
echo "Architectures: ResUNet, Attention ResUNet"
echo "Dataset: ./dataset_full_stack/ (100 images, 1,980 patches)"
echo "Image Size: 256×256"
echo ""
echo "Search Space:"
echo "  Learning Rate: [1e-5, 2e-5, 5e-5]"
echo "  Dropout: [0.3, 0.4, 0.5]"
echo "  Batch Size: [4, 8]"
echo ""
echo "Fixed Parameters:"
echo "  Filters: 64"
echo "  Loss: combined (0.7×Dice + 0.3×Focal)"
echo "  CV Folds: 3 (for efficiency)"
echo "  Max Epochs: 25"
echo "  Early Stopping Patience: 8"
echo ""
echo "Total Configurations: 2 arch × 3 LR × 3 dropout × 2 batch = 36"
echo "Total Models: 36 configs × 3 folds = 108 models"
echo ""
echo "Expected Runtime: 8-12 hours"
echo "=============================="
echo ""

# =======================================================================
# ENVIRONMENT SETUP
# =======================================================================

echo "=== ENVIRONMENT SETUP ==="

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

echo "✓ TensorFlow Container: $image"
echo "=========================="
echo ""

# =======================================================================
# PRE-EXECUTION CHECKS
# =======================================================================

echo "=== PRE-EXECUTION CHECKS ==="

# Check Python script exists
echo "1. Checking Python scripts..."
if [ ! -f "./hyperparameter_search_residual_architectures.py" ]; then
    echo "   ✗ ERROR: hyperparameter_search_residual_architectures.py not found!"
    echo "   Current directory: $(pwd)"
    ls -la ./*.py | head -10
    exit 1
fi
echo "   ✓ hyperparameter_search_residual_architectures.py found"

# Check required modules exist
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

# Check dataset structure
echo ""
echo "2. Checking dataset structure..."
if [ ! -d "./dataset_full_stack/images/" ] || [ ! -d "./dataset_full_stack/masks/" ]; then
    echo "   ✗ ERROR: Dataset directories not found!"
    echo "   Expected: ./dataset_full_stack/images/ and ./dataset_full_stack/masks/"
    echo "   Current directory contents:"
    ls -la ./
    exit 1
fi

# Count files
img_count=$(find ./dataset_full_stack/images/ -name "*.tif" 2>/dev/null | wc -l)
mask_count=$(find ./dataset_full_stack/masks/ -name "*.tif" 2>/dev/null | wc -l)

echo "   ✓ Dataset directories found"
echo "   ✓ Images found: $img_count .tif files"
echo "   ✓ Masks found: $mask_count .tif files"

if [ $img_count -eq 0 ] || [ $mask_count -eq 0 ]; then
    echo "   ✗ ERROR: No .tif files found in dataset directories"
    exit 1
fi

if [ $img_count -ne $mask_count ]; then
    echo "   ⚠ WARNING: Unequal number of images ($img_count) and masks ($mask_count)"
fi

echo "=============================="
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
        print('Memory growth setting (already initialized):', e)

# Test basic GPU operation
if gpus:
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
    print('⚠ WARNING: No GPU detected! Training will be very slow.')

# Check key dependencies
print()
print('Checking dependencies:')
deps_to_check = ['cv2', 'PIL', 'matplotlib', 'numpy', 'sklearn', 'pandas', 'keras']
missing_deps = []
for dep in deps_to_check:
    try:
        __import__(dep)
        print(f'  ✓ {dep}')
    except ImportError:
        print(f'  ✗ {dep} - MISSING!')
        missing_deps.append(dep)

if missing_deps:
    print()
    print(f'ERROR: Missing dependencies: {missing_deps}')
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
# CREATE TIMESTAMPED OUTPUT DIRECTORY
# =======================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="hyperparameter_search_${TIMESTAMP}"

echo "=== OUTPUT CONFIGURATION ==="
echo "Results directory: $OUTPUT_DIR"
echo "This will contain:"
echo "  - hyperparameter_search_summary.json (best configs)"
echo "  - [config_name]/fold_[1-3]/ (individual results)"
echo "=============================="
echo ""

# =======================================================================
# EXECUTE HYPERPARAMETER SEARCH
# =======================================================================

echo "🚀 STARTING HYPERPARAMETER SEARCH"
echo "=============================================="
echo "Searching for optimal hyperparameters"
echo "Target performance: 69.94% Jaccard (U-Net baseline)"
echo ""
echo "Search strategy:"
echo "  - Lower learning rates (1e-5, 2e-5) to fix ResUNet collapse"
echo "  - Higher dropout (0.4, 0.5) for better regularization"
echo "  - Different batch sizes (4, 8) for gradient noise tuning"
echo ""
echo "Progress tracking:"
echo "  Each configuration tests 3-fold CV"
echo "  36 configurations × 3 folds = 108 models"
echo "  ~5-7 minutes per model"
echo "  Total: 8-12 hours"
echo ""
echo "=============================================="
echo ""

# Execute with logging
singularity exec --nv "$image" python3 hyperparameter_search_residual_architectures.py 2>&1 | tee "hyperparam_search_console_${TIMESTAMP}.log"

# Capture exit code
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

    # Check if output directory was created
    if [ -d "$OUTPUT_DIR" ]; then
        echo "📁 RESULTS DIRECTORY: $OUTPUT_DIR"
        echo ""

        # Summary of generated files
        echo "Generated files:"
        echo "-------------------------"
        find "$OUTPUT_DIR" -name "hyperparameter_search_summary.json" -exec echo "  ✓ {}" \;

        config_count=$(find "$OUTPUT_DIR" -maxdepth 1 -type d | wc -l)
        model_count=$(find "$OUTPUT_DIR" -name "best_model.keras" 2>/dev/null | wc -l)

        echo ""
        echo "📊 SEARCH SUMMARY:"
        echo "  Configurations tested: $((config_count - 1))"  # Subtract parent dir
        echo "  Models trained: $model_count"

        # Check for summary file and display quick results
        if [ -f "$OUTPUT_DIR/hyperparameter_search_summary.json" ]; then
            echo ""
            echo "📈 QUICK RESULTS PREVIEW:"
            echo "========================"

            singularity exec --nv "$image" python3 -c "
import json
import sys

try:
    with open('$OUTPUT_DIR/hyperparameter_search_summary.json', 'r') as f:
        summary = json.load(f)

    best_configs = summary.get('best_configs', {})

    print('\\nBest Configurations Found:')
    print('-' * 60)

    for arch, config_data in best_configs.items():
        perf = config_data.get('mean_best_jacard', 0)
        std = config_data.get('std_best_jacard', 0)
        config = config_data.get('config', {})

        print(f'\\n{arch.upper()}:')
        print(f'  Performance: {perf:.4f} ± {std:.4f}')
        print(f'  Learning Rate: {config.get(\"learning_rate\", \"N/A\")}')
        print(f'  Dropout: {config.get(\"dropout\", \"N/A\")}')
        print(f'  Batch Size: {config.get(\"batch_size\", \"N/A\")}')

        # Compare to baseline
        baseline = 0.6994
        improvement = ((perf - baseline) / baseline) * 100

        if improvement > 0:
            print(f'  vs U-Net: {improvement:+.1f}% ✅ IMPROVED!')
        elif improvement > -5:
            print(f'  vs U-Net: {improvement:+.1f}% (within 5%)')
        else:
            print(f'  vs U-Net: {improvement:+.1f}% (still underperforming)')

    print()
    print('-' * 60)
    print('Full results in hyperparameter_search_summary.json')

except Exception as e:
    print(f'Unable to parse summary: {e}')
    print('Results saved but summary parsing failed.')
" 2>/dev/null || echo "Results saved. Check summary file for details."

        else
            echo "⚠ Summary file not found (may indicate incomplete run)"
        fi

    else
        echo "⚠ WARNING: Expected output directory not found!"
        echo "Searching for hyperparameter_search directories..."
        find . -maxdepth 1 -type d -name "hyperparameter_search_*" -mtime -1
    fi

    echo ""
    echo "🎯 NEXT STEPS:"
    echo "============="
    echo "1. Review best configurations in:"
    echo "   $OUTPUT_DIR/hyperparameter_search_summary.json"
    echo ""
    echo "2. If improvement found:"
    echo "   - Re-run full 5-fold CV with best config"
    echo "   - Compare to baseline U-Net statistically"
    echo ""
    echo "3. If no improvement:"
    echo "   - Continue using U-Net (69.94% Jaccard)"
    echo "   - Consider alternative architectures or data improvements"

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Hyperparameter search failed!"
    echo ""
    echo "🔍 ERROR ANALYSIS:"
    echo "=================="
    echo "Check the console log for detailed error information:"
    echo "   hyperparam_search_console_${TIMESTAMP}.log"
    echo ""

    if [ -f "hyperparam_search_console_${TIMESTAMP}.log" ]; then
        echo "Last 30 lines of console log:"
        echo "-----------------------------"
        tail -30 "hyperparam_search_console_${TIMESTAMP}.log"
    fi
    echo ""

    echo "🔧 COMMON ISSUES AND SOLUTIONS:"
    echo "==============================="
    echo "1. Dataset path issues: Ensure ./dataset_full_stack/ exists"
    echo "2. GPU memory issues: Reduce max batch_size in search space"
    echo "3. Import errors: Check model_architectures.py and loss_functions_fixed.py"
    echo "4. Timeout: Job may need more than 12 hours (increase walltime)"
    echo ""
fi

echo ""
echo "📝 CONSOLE LOG: hyperparam_search_console_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "HYPERPARAMETER SEARCH JOB COMPLETE"
echo "Architectures: ResUNet, Attention ResUNet"
echo "Method: Grid search (LR, dropout, batch size)"
echo "Framework: TensorFlow/Keras"
echo "======================================="
