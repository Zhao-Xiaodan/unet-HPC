#!/bin/bash
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Architecture_Comparison
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# ARCHITECTURE COMPARISON STUDY - PBS SCRIPT
# =======================================================================
# Comparing U-Net, ResUNet, and Attention ResUNet for microbead segmentation
# Using 5-fold cross-validation for reliable performance estimates
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "ARCHITECTURE COMPARISON STUDY"
echo "======================================================================="
echo "Architectures: U-Net, ResUNet, Attention ResUNet"
echo "Task: Microbead segmentation"
echo "Evaluation: 5-fold cross-validation"
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

echo "=== STUDY CONFIGURATION ==="
echo "Dataset Images: ./dataset_full_stack/images/ (100 images)"
echo "Dataset Masks: ./dataset_full_stack/masks/ (100 masks)"
echo "Image Size: 256×256 (resized from 512×512)"
echo "Batch Size: 4"
echo "Epochs: 20 (with early stopping, patience=5)"
echo "CV Folds: 5"
echo "Total Models: 15 (3 architectures × 5 folds)"
echo ""
echo "Architectures:"
echo "  1. U-Net (baseline: 60.97% ± 11.5% Jaccard)"
echo "  2. ResUNet (with residual connections)"
echo "  3. Attention ResUNet (with attention gates)"
echo ""
echo "Hyperparameters (same for all):"
echo "  - filters: 64"
echo "  - dropout: 0.3"
echo "  - loss: combined (0.7×Dice + 0.3×Focal)"
echo "  - optimizer: Adam (lr=5e-5, clipnorm=1.0)"
echo ""
echo "Expected Runtime: 2.5-3.5 hours"
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
if [ ! -f "./validate_architecture_comparison.py" ]; then
    echo "   ✗ ERROR: validate_architecture_comparison.py not found!"
    echo "   Current directory: $(pwd)"
    ls -la ./*.py | head -10
    exit 1
fi
echo "   ✓ validate_architecture_comparison.py found"

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

if [ $img_count -lt 50 ]; then
    echo "   ⚠ WARNING: Only $img_count images found. Expected ~100 for full dataset."
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
import os

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
        print(f'  Result shape: {c.shape}')
    except Exception as e:
        print('✗ GPU operation failed:', e)
        sys.exit(1)
else:
    print('⚠ WARNING: No GPU detected! Training will be very slow.')

# Check key dependencies
print()
print('Checking dependencies:')
deps_to_check = [
    'cv2', 'PIL', 'matplotlib', 'numpy', 'sklearn',
    'pandas', 'keras', 'json'
]
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
OUTPUT_DIR="validation_arch_comparison_${TIMESTAMP}"

echo "=== OUTPUT CONFIGURATION ==="
echo "Results directory: $OUTPUT_DIR"
echo "This will contain:"
echo "  - architecture_comparison_summary.json"
echo "  - unet/fold_1/...fold_5/"
echo "  - resunet/fold_1/...fold_5/"
echo "  - attention_resunet/fold_1/...fold_5/"
echo "=============================="
echo ""

# =======================================================================
# EXECUTE ARCHITECTURE COMPARISON
# =======================================================================

echo "🚀 STARTING ARCHITECTURE COMPARISON STUDY"
echo "=============================================="
echo "Testing 3 architectures with 5-fold CV each"
echo "Total: 15 models to train"
echo ""
echo "Progress tracking:"
echo "  Each fold takes ~8-12 minutes"
echo "  Each architecture takes ~40-60 minutes"
echo "  Total time: ~2.5-3.5 hours"
echo ""
echo "Training order:"
echo "  1. U-Net (folds 1-5)"
echo "  2. ResUNet (folds 1-5)"
echo "  3. Attention ResUNet (folds 1-5)"
echo "=============================================="
echo ""

# Execute with logging
singularity exec --nv "$image" python3 validate_architecture_comparison.py 2>&1 | tee "arch_comparison_console_${TIMESTAMP}.log"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "ARCHITECTURE COMPARISON STUDY COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✓ SUCCESS"
    echo ""
    echo "✓ Architecture comparison completed successfully!"
    echo ""

    # Check if output directory was created
    if [ -d "$OUTPUT_DIR" ]; then
        echo "📁 RESULTS DIRECTORY: $OUTPUT_DIR"
        echo ""

        # Summary of generated files
        echo "Generated files structure:"
        echo "-------------------------"
        tree -L 2 "$OUTPUT_DIR" 2>/dev/null || find "$OUTPUT_DIR" -maxdepth 2 -type f | sort
        echo ""

        # Check for summary file
        if [ -f "$OUTPUT_DIR/architecture_comparison_summary.json" ]; then
            echo "✓ Summary file generated"
            echo ""
            echo "📊 QUICK RESULTS PREVIEW:"
            echo "========================"

            singularity exec --nv "$image" python3 -c "
import json
import sys

try:
    with open('$OUTPUT_DIR/architecture_comparison_summary.json', 'r') as f:
        summary = json.load(f)

    comparison = summary['comparison']

    print('\\nPerformance Summary (Best Val Jaccard):')
    print('-' * 60)

    # Display each architecture
    for arch in ['unet', 'resunet', 'attention_resunet']:
        if arch in comparison:
            stats = comparison[arch]
            mean = stats['best_val_jacard']['mean']
            std = stats['best_val_jacard']['std']

            arch_name = arch.upper().replace('_', ' ')
            print(f'{arch_name:20s}: {mean:.4f} ± {std:.4f}')

    print()
    print('Baseline Reference:')
    baseline = summary.get('baseline_reference', {})
    if baseline:
        print(f'  U-Net CV (previous): {baseline.get(\"unet_cv_mean\", 0):.4f}')

    print()
    print('-' * 60)
    print('✓ Run analysis script for detailed results and visualizations:')
    print(f'  python analyze_architecture_comparison.py $OUTPUT_DIR')

except Exception as e:
    print(f'Unable to parse summary: {e}')
    print('Results saved but summary parsing failed.')
" 2>/dev/null || echo "Results saved. Run analysis script for details."

        else
            echo "⚠ Summary file not found (may indicate incomplete run)"
        fi

        echo ""
        echo "========================"
        echo ""

        # Check model files
        model_count=$(find "$OUTPUT_DIR" -name "best_model.keras" 2>/dev/null | wc -l)
        history_count=$(find "$OUTPUT_DIR" -name "history.csv" 2>/dev/null | wc -l)

        echo "📈 FILES GENERATED:"
        echo "  Model files (.keras): $model_count (expected 15)"
        echo "  History files (.csv): $history_count (expected 15)"

        if [ $model_count -lt 15 ] || [ $history_count -lt 15 ]; then
            echo ""
            echo "  ⚠ WARNING: Fewer files than expected!"
            echo "  Some folds may have failed or training was interrupted."
        fi

    else
        echo "⚠ WARNING: Expected output directory not found!"
        echo "Training may have failed or used different directory."
        echo "Searching for validation_arch_comparison directories..."
        find . -maxdepth 1 -type d -name "validation_arch_comparison_*" -mtime -1
    fi

    echo ""
    echo "🎯 NEXT STEPS:"
    echo "============="
    echo "1. Run analysis script to generate visualizations and report:"
    echo "   python analyze_architecture_comparison.py $OUTPUT_DIR"
    echo ""
    echo "2. Check the generated report:"
    echo "   cat $OUTPUT_DIR/ARCHITECTURE_COMPARISON_REPORT.md"
    echo ""
    echo "3. Review visualizations:"
    echo "   - architecture_performance_comparison.png"
    echo "   - architecture_training_curves.png"
    echo "   - architecture_convergence_analysis.png"

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Architecture comparison failed!"
    echo ""
    echo "🔍 ERROR ANALYSIS:"
    echo "=================="
    echo "Check the console log for detailed error information:"
    echo "   arch_comparison_console_${TIMESTAMP}.log"
    echo ""

    if [ -f "arch_comparison_console_${TIMESTAMP}.log" ]; then
        echo "Last 30 lines of console log:"
        echo "-----------------------------"
        tail -30 "arch_comparison_console_${TIMESTAMP}.log"
    fi
    echo ""

    echo "🔧 COMMON ISSUES AND SOLUTIONS:"
    echo "==============================="
    echo "1. Dataset path issues: Ensure ./dataset_full_stack/images/ and ./dataset_full_stack/masks/ exist"
    echo "2. Missing dependencies: Check that all required packages are installed"
    echo "3. GPU memory issues: Reduce batch_size in BASE_CONFIG (currently 4)"
    echo "4. Import errors: Ensure model_architectures.py and loss_functions_fixed.py are present"
    echo "5. Insufficient walltime: Job may need more than 6 hours if slow"
    echo ""
    echo "Check the Python script for errors:"
    echo "  python -m py_compile validate_architecture_comparison.py"
fi

echo ""
echo "📝 CONSOLE LOG: arch_comparison_console_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "ARCHITECTURE COMPARISON JOB COMPLETE"
echo "Architectures: U-Net, ResUNet, Attention ResUNet"
echo "Method: 5-fold cross-validation"
echo "Framework: TensorFlow/Keras"
echo "======================================="
