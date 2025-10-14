#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oed
#PBS -N AttResUNet_Search
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# ATTENTION RESUNET HYPERPARAMETER SEARCH - PBS SCRIPT
# =======================================================================
# Based on ResUNet search results (hyperparameter_search_20251013_154754)
# Goal: Test if attention gates improve ResUNet performance
# Target: >55% Jaccard to justify further investigation
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "ATTENTION RESUNET HYPERPARAMETER SEARCH"
echo "======================================================================="
echo "Based on: ResUNet search (42.40% Jaccard achieved)"
echo "Goal: Test if attention gates can exceed 55% Jaccard"
echo "Method: Focused search around optimal ResUNet hyperparameters"
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
echo "Architecture: Attention ResUNet"
echo "Dataset: ./dataset_full_stack/ (100 images, 1,980 patches)"
echo "Image Size: 256×256"
echo ""
echo "Search Space (focused based on ResUNet findings):"
echo "  Learning Rate: [1.5e-5, 2e-5, 2.5e-5] ← centered on ResUNet optimum"
echo "  Dropout: [0.2, 0.3, 0.4] ← lower range (higher dropout hurt)"
echo "  Batch Size: [4, 8]"
echo ""
echo "Fixed Parameters:"
echo "  Filters: 64"
echo "  Loss: combined (0.7×Dice + 0.3×Focal)"
echo "  CV Folds: 3 (for efficiency)"
echo "  Max Epochs: 30"
echo "  Early Stopping Patience: 8"
echo ""
echo "Total Configurations: 3 LR × 3 dropout × 2 batch = 18"
echo "Total Models: 18 configs × 3 folds = 54 models"
echo ""
echo "Baselines for Comparison:"
echo "  U-Net:               69.94% Jaccard"
echo "  ResUNet (baseline):  39.95% Jaccard"
echo "  ResUNet (optimized): 42.40% Jaccard"
echo ""
echo "Expected Runtime: 6-10 hours (more efficient search space)"
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
echo "==========================="
echo ""

# =======================================================================
# PRE-EXECUTION CHECKS
# =======================================================================

echo "=== PRE-EXECUTION CHECKS ==="

# Check Python script exists
echo "1. Checking Python scripts..."
if [ ! -f "./hyperparameter_search_attention_resunet.py" ]; then
    echo "   ✗ ERROR: hyperparameter_search_attention_resunet.py not found!"
    echo "   Current directory: $(pwd)"
    ls -la ./*.py | head -10
    exit 1
fi
echo "   ✓ hyperparameter_search_attention_resunet.py found"

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
OUTPUT_DIR="attention_resunet_search_${TIMESTAMP}"

echo "=== OUTPUT CONFIGURATION ==="
echo "Results directory: $OUTPUT_DIR"
echo "This will contain:"
echo "  - attention_resunet_search_summary.json (best configs)"
echo "  - [config_name]/fold_[1-3]/ (individual results)"
echo "=============================="
echo ""

# =======================================================================
# EXECUTE HYPERPARAMETER SEARCH
# =======================================================================

echo "🚀 STARTING ATTENTION RESUNET HYPERPARAMETER SEARCH"
echo "=============================================="
echo "Testing if attention gates improve ResUNet performance"
echo ""
echo "Strategy:"
echo "  - Focus learning rate around 2e-05 (ResUNet optimum)"
echo "  - Use lower dropout (0.2-0.4) since higher dropout hurt"
echo "  - Test both batch sizes (4, 8)"
echo ""
echo "Decision criteria:"
echo "  >69.94%: SUCCESS - Beats U-Net!"
echo "  >55.00%: PROMISING - Run full 5-fold validation"
echo "  >42.40%: MARGINAL - Small improvement over ResUNet"
echo "  ≤42.40%: FAILURE - Attention gates don't help"
echo ""
echo "Progress tracking:"
echo "  Each configuration tests 3-fold CV"
echo "  18 configurations × 3 folds = 54 models"
echo "  ~5-7 minutes per model"
echo "  Total: 6-10 hours"
echo ""
echo "=============================================="
echo ""

# Execute with logging
singularity exec --nv "$image" python3 hyperparameter_search_attention_resunet.py 2>&1 | tee "attention_resunet_search_console_${TIMESTAMP}.log"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "ATTENTION RESUNET HYPERPARAMETER SEARCH COMPLETED"
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
        find "$OUTPUT_DIR" -name "attention_resunet_search_summary.json" -exec echo "  ✓ {}" \;

        config_count=$(find "$OUTPUT_DIR" -maxdepth 1 -type d | wc -l)
        model_count=$(find "$OUTPUT_DIR" -name "best_model.keras" 2>/dev/null | wc -l)

        echo ""
        echo "📊 SEARCH SUMMARY:"
        echo "  Configurations tested: $((config_count - 1))"  # Subtract parent dir
        echo "  Models trained: $model_count"

        # Check for summary file and display quick results
        if [ -f "$OUTPUT_DIR/attention_resunet_search_summary.json" ]; then
            echo ""
            echo "📈 QUICK RESULTS PREVIEW:"
            echo "========================"

            singularity exec --nv "$image" python3 -c "
import json
import sys

try:
    with open('$OUTPUT_DIR/attention_resunet_search_summary.json', 'r') as f:
        summary = json.load(f)

    best_config = summary.get('best_config', {})
    best_perf = best_config.get('mean_best_jacard', 0)
    std = best_config.get('std_best_jacard', 0)
    config = best_config.get('config', {})

    print()
    print('BEST CONFIGURATION FOUND:')
    print('-' * 60)
    print(f'Performance: {best_perf:.4f} ± {std:.4f}')
    print(f'Learning Rate: {config.get(\"learning_rate\", \"N/A\")}')
    print(f'Dropout: {config.get(\"dropout\", \"N/A\")}')
    print(f'Batch Size: {config.get(\"batch_size\", \"N/A\")}')
    print()

    # Compare to baselines
    baselines = summary.get('baselines', {})
    unet_baseline = baselines.get('unet', 0.6994)
    resunet_baseline = baselines.get('resunet_baseline', 0.3995)
    resunet_optimized = baselines.get('resunet_optimized', 0.4240)

    print('COMPARISON:')
    print('-' * 60)
    print(f'U-Net (baseline):     {unet_baseline:.4f}')
    print(f'ResUNet (baseline):   {resunet_baseline:.4f}')
    print(f'ResUNet (optimized):  {resunet_optimized:.4f}')
    print(f'Attention ResUNet:    {best_perf:.4f} ← THIS RESULT')
    print()

    # Decision
    if best_perf > unet_baseline:
        print('✅ SUCCESS! Attention ResUNet EXCEEDS U-Net!')
        print('   → Use Attention ResUNet as primary architecture')
    elif best_perf > 0.55:
        print('✅ PROMISING! Attention gates improve performance.')
        print('   → Run full 5-fold validation to confirm')
    elif best_perf > resunet_optimized:
        print('⚠️  MARGINAL: Small improvement over ResUNet.')
        print('   → Decide based on accuracy vs computational cost')
    else:
        print('❌ FAILURE: Attention gates do not help.')
        print('   → Abandon residual architectures, stick with U-Net')

    print()
    print('-' * 60)
    print('Full results in attention_resunet_search_summary.json')

except Exception as e:
    print(f'Unable to parse summary: {e}')
    print('Results saved but summary parsing failed.')
" 2>/dev/null || echo "Results saved. Check summary file for details."

        else
            echo "⚠ Summary file not found (may indicate incomplete run)"
        fi

    else
        echo "⚠ WARNING: Expected output directory not found!"
        echo "Searching for attention_resunet_search directories..."
        find . -maxdepth 1 -type d -name "attention_resunet_search_*" -mtime -1
    fi

    echo ""
    echo "🎯 NEXT STEPS:"
    echo "============="
    echo "1. Review best configuration in:"
    echo "   $OUTPUT_DIR/attention_resunet_search_summary.json"
    echo ""
    echo "2. If performance >55% Jaccard:"
    echo "   - Run full 5-fold CV with best config"
    echo "   - Compare to U-Net statistically"
    echo ""
    echo "3. If performance ≤55% Jaccard:"
    echo "   - Abandon residual architectures"
    echo "   - Stick with U-Net (69.94% Jaccard)"
    echo "   - Focus on data improvements instead"

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Hyperparameter search failed!"
    echo ""
    echo "🔍 ERROR ANALYSIS:"
    echo "=================="
    echo "Check the console log for detailed error information:"
    echo "   attention_resunet_search_console_${TIMESTAMP}.log"
    echo ""

    if [ -f "attention_resunet_search_console_${TIMESTAMP}.log" ]; then
        echo "Last 30 lines of console log:"
        echo "-----------------------------"
        tail -30 "attention_resunet_search_console_${TIMESTAMP}.log"
    fi
    echo ""

    echo "🔧 COMMON ISSUES AND SOLUTIONS:"
    echo "==============================="
    echo "1. Dataset path issues: Ensure ./dataset_full_stack/ exists"
    echo "2. GPU memory issues: Reduce max batch_size in search space"
    echo "3. Import errors: Check model_architectures.py has attention_resunet"
    echo "4. Timeout: Job may need more than 12 hours (increase walltime)"
    echo ""
fi

echo ""
echo "📝 CONSOLE LOG: attention_resunet_search_console_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "ATTENTION RESUNET SEARCH JOB COMPLETE"
echo "Method: Focused search around ResUNet optimum"
echo "Framework: TensorFlow/Keras"
echo "======================================="
