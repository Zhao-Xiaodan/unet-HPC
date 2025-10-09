#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Microbead_Optimized_Training
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# OPTIMIZED MICROBEAD SEGMENTATION TRAINING - HPC PBS SCRIPT
# =======================================================================
# Training with corrected hyperparameters for dense microbead segmentation
# Based on dataset analysis: 109 objects/image (36× more than mitochondria)
#
# Key changes from mitochondria training:
# - Learning Rate: 1e-4 (was 1e-3) - Account for 36× stronger gradients
# - Batch Size: 32 (was 8-16) - Better gradient stability
# - Dropout: 0.3 (was 0.0) - Prevent overfitting dense objects
# - Loss: Dice (was Focal) - Direct overlap optimization
# - Stratified split by object density
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "OPTIMIZED MICROBEAD SEGMENTATION TRAINING"
echo "======================================================================="
echo "Dataset analysis findings:"
echo "  - 73 images with 109.4 objects/image average (36× more than mitochondria)"
echo "  - Requires corrected hyperparameters for dense object segmentation"
echo ""
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $(free -h | grep Mem | awk '{print $2}'), CPUs: $(nproc)"
echo ""

# =======================================================================
# TRAINING CONFIGURATION SUMMARY
# =======================================================================

echo "=== OPTIMIZED HYPERPARAMETERS ==="
echo "Comparison with mitochondria training:"
echo ""
echo "Learning Rate:"
echo "  Mitochondria: 1e-3 (UNet), 1e-4 (Attention)"
echo "  Microbeads:   1e-4 (all models) ← 10× LOWER for dense objects"
echo ""
echo "Batch Size:"
echo "  Mitochondria: 8-16"
echo "  Microbeads:   32 ← 2-4× LARGER for gradient stability"
echo ""
echo "Dropout Regularization:"
echo "  Mitochondria: 0.0 (minimal)"
echo "  Microbeads:   0.3 ← Prevent overfitting uniform objects"
echo ""
echo "Loss Function:"
echo "  Mitochondria: Binary Focal Loss (γ=2)"
echo "  Microbeads:   Dice Loss ← Direct overlap optimization"
echo ""
echo "Validation Split:"
echo "  Mitochondria: Random 90/10"
echo "  Microbeads:   Stratified 85/15 by object density"
echo ""
echo "Expected Results:"
echo "  Previous (wrong params): Val Jaccard 0.14 → 0.0 (collapsed)"
echo "  Expected (optimized):    Val Jaccard 0.50-0.70 (stable)"
echo "=================================="
echo ""

# =======================================================================
# ENVIRONMENT SETUP
# =======================================================================

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

# Check 1: Dataset directory
echo "1. Checking dataset..."
if [ -d "./dataset_microscope/images" ] && [ -d "./dataset_microscope/masks" ]; then
    echo "   ✓ Dataset directories found"

    img_count=$(find ./dataset_microscope/images/ -type f \( -name "*.tif" -o -name "*.png" -o -name "*.jpg" \) | wc -l)
    mask_count=$(find ./dataset_microscope/masks/ -type f \( -name "*.tif" -o -name "*.png" -o -name "*.jpg" \) | wc -l)

    echo "   ✓ Images: $img_count files"
    echo "   ✓ Masks: $mask_count files"

    if [ $img_count -ne $mask_count ]; then
        echo "   WARNING: Image/mask count mismatch ($img_count vs $mask_count)"
    fi

    if [ $img_count -lt 50 ]; then
        echo "   WARNING: Small dataset ($img_count images). Results may vary."
    fi
else
    echo "   ERROR: Dataset not found!"
    echo "   Expected: ./dataset_microscope/images/ and ./dataset_microscope/masks/"
    ls -la ./dataset_microscope/ 2>/dev/null || echo "   dataset_microscope/ directory missing"
    exit 1
fi

# Check 2: Training script
echo ""
echo "2. Checking training script..."
if [ -f "./train_microbead_optimized.py" ]; then
    echo "   ✓ Optimized training script found"
else
    echo "   ERROR: train_microbead_optimized.py not found"
    echo "   Please ensure the file is uploaded to HPC"
    ls -la ./*microbead*.py 2>/dev/null
    exit 1
fi

# Check 3: Model definitions
echo ""
echo "3. Checking model files..."
if [ -f "./models.py" ] || [ -f "./224_225_226_models.py" ]; then
    echo "   ✓ Model definitions found"

    if [ -f "./224_225_226_models.py" ] && [ ! -f "./models.py" ]; then
        echo "   ✓ Creating models.py from 224_225_226_models.py"
        cp ./224_225_226_models.py ./models.py
    fi
else
    echo "   ERROR: Model definition file not found"
    echo "   Expected: models.py or 224_225_226_models.py"
    exit 1
fi

# Check 4: Analysis results (optional but recommended)
echo ""
echo "4. Checking dataset analysis..."
if [ -f "./dataset_analysis/summary.json" ]; then
    echo "   ✓ Dataset analysis results found"
    echo "   Analysis summary:"
    cat ./dataset_analysis/summary.json | grep -E "mean_objects_per_image|mean_positive_ratio" || true
else
    echo "   ℹ  No dataset analysis found (optional)"
    echo "   Recommended: Run analyze_microbead_dataset.py first"
fi

echo "============================"
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
else:
    print('WARNING: No GPUs detected - training will be SLOW on CPU')

# Check dependencies
print()
print('Checking dependencies:')
deps = ['cv2', 'PIL', 'matplotlib', 'numpy', 'sklearn', 'pandas']
for dep in deps:
    try:
        __import__(dep)
        print(f'  ✓ {dep}')
    except ImportError:
        print(f'  ✗ {dep} - MISSING!')

print()
print('✓ Environment ready')
"
echo "==============================="
echo ""

# =======================================================================
# CREATE OUTPUT DIRECTORY
# =======================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="microbead_training_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Created output directory: $OUTPUT_DIR"
echo ""

# =======================================================================
# RUN OPTIMIZED TRAINING
# =======================================================================

echo "🚀 STARTING OPTIMIZED MICROBEAD TRAINING"
echo "=============================================="
echo "Training 3 models with corrected hyperparameters:"
echo "  1. UNet (LR=1e-4, BS=32, Dropout=0.3)"
echo "  2. Attention UNet (LR=1e-4, BS=32, Dropout=0.3)"
echo "  3. Attention ResUNet (LR=1e-4, BS=32, Dropout=0.3)"
echo ""
echo "Expected training time: 6-10 hours (with early stopping)"
echo "Expected validation Jaccard: 0.50-0.70"
echo ""
echo "This will REPLACE the previous failed training!"
echo "=============================================="
echo ""

# Execute training
singularity exec --nv "$image" python3 train_microbead_optimized.py 2>&1 | tee "${OUTPUT_DIR}/training_console.log"

# Capture exit code
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

    # Find output directory (script creates timestamped dir)
    SCRIPT_OUTPUT=$(ls -td microbead_training_* 2>/dev/null | head -1)

    if [ -n "$SCRIPT_OUTPUT" ] && [ -d "$SCRIPT_OUTPUT" ]; then
        echo "📊 TRAINING RESULTS:"
        echo "==================="

        # Display training summary if available
        if [ -f "$SCRIPT_OUTPUT/training_summary.csv" ]; then
            echo "Performance summary:"
            cat "$SCRIPT_OUTPUT/training_summary.csv"
            echo ""
        fi

        # Count generated files
        model_count=$(find "$SCRIPT_OUTPUT" -name "*.hdf5" | wc -l)
        csv_count=$(find "$SCRIPT_OUTPUT" -name "*.csv" | wc -l)

        echo "Generated files:"
        echo "  - Model files (.hdf5): $model_count"
        echo "  - Training histories (.csv): $csv_count"
        echo ""

        echo "📁 OUTPUT DIRECTORY: $SCRIPT_OUTPUT"
        echo ""

        # Check if improvement achieved
        echo "🔍 PERFORMANCE CHECK:"
        echo "===================="

        singularity exec --nv "$image" python3 -c "
import pandas as pd
import os

output_dir = '$SCRIPT_OUTPUT'
summary_file = os.path.join(output_dir, 'training_summary.csv')

if os.path.exists(summary_file):
    df = pd.read_csv(summary_file)
    best_jaccard = df['best_val_jacard'].max()
    best_model = df.loc[df['best_val_jacard'].idxmax(), 'model']

    print(f'Best model: {best_model}')
    print(f'Best validation Jaccard: {best_jaccard:.4f}')
    print()

    # Compare with previous failed training
    previous_best = 0.1427  # Attention ResUNet from mitochondria params

    print('Comparison with previous training:')
    print(f'  Previous (mitochondria params): 0.1427 → 0.0 (collapsed)')
    print(f'  Current (microbead params):     {best_jaccard:.4f}')
    print()

    if best_jaccard > 0.50:
        improvement = best_jaccard / previous_best
        print(f'  ✓✓ EXCELLENT: {improvement:.1f}× improvement!')
        print(f'  → Validation Jaccard > 0.50 - Production ready!')
    elif best_jaccard > 0.30:
        improvement = best_jaccard / previous_best
        print(f'  ✓ GOOD: {improvement:.1f}× improvement')
        print(f'  → Significant progress, may benefit from fine-tuning')
    elif best_jaccard > 0.20:
        improvement = best_jaccard / previous_best
        print(f'  ⚠  MODERATE: {improvement:.1f}× improvement')
        print(f'  → Some progress, check training curves')
    else:
        print(f'  ✗ LIMITED: Further investigation needed')
        print(f'  → Check dataset quality and training logs')
else:
    print('Summary file not found')
" 2>/dev/null || echo "Could not analyze results"

        echo ""
        echo "📥 TO DOWNLOAD RESULTS:"
        echo "======================="
        echo "From your local machine:"
        echo "  scp -r phyzxi@hpc:~/scratch/unet-HPC/$SCRIPT_OUTPUT ./local_results/"
        echo ""
    fi

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Training failed!"
    echo ""

    echo "🔍 ERROR ANALYSIS:"
    echo "=================="

    if [ -f "${OUTPUT_DIR}/training_console.log" ]; then
        echo "Last 30 lines of training log:"
        tail -30 "${OUTPUT_DIR}/training_console.log"
    fi
    echo ""

    echo "🔧 COMMON ISSUES:"
    echo "================="
    echo "1. Dataset path: Verify dataset_microscope/images/ and masks/ exist"
    echo "2. Memory: Try reducing batch size from 32 to 16 in script"
    echo "3. Dependencies: Check all packages installed in container"
    echo "4. Model files: Ensure models.py is present"
fi

echo ""
echo "📝 FULL LOGS:"
echo "============="
echo "  Training console: ${OUTPUT_DIR}/training_console.log"
echo "  PBS output: Microbead_Optimized_Training.o${PBS_JOBID##*.}"
echo ""
echo "======================================================================="
echo "MICROBEAD OPTIMIZED TRAINING - JOB COMPLETE"
echo "======================================================================="
echo "Models: UNet, Attention UNet, Attention ResUNet"
echo "Hyperparameters: Optimized for 109 objects/image (36× mitochondria)"
echo "Expected: Val Jaccard 0.50-0.70 (vs previous 0.14→0.0)"
echo "======================================================================="
