#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Microbead_Hyperparam_Search
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# HYPERPARAMETER SEARCH - MICROBEAD SEGMENTATION
# =======================================================================
# Systematic search to find optimal hyperparameters through experiments
#
# Search space:
# - Image Size: 512×512 (original resolution, NOT resized to 256×256)
# - Learning Rate: [5e-5, 1e-4, 2e-4]
# - Batch Size: [4, 8, 16] (reduced for 512×512 - 4× memory vs 256×256)
# - Dropout: [0.0, 0.1, 0.2, 0.3]
# - Loss: [dice, focal, combined]
#
# Total combinations:
#   Grid search: 3 × 3 × 4 × 3 = 108 experiments (~12-24 hours, slower with 512×512)
#   Random search: 20 experiments (~3-5 hours)
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "HYPERPARAMETER SEARCH - MICROBEAD SEGMENTATION"
echo "======================================================================="
echo ""
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $(free -h | grep Mem | awk '{print $2}'), CPUs: $(nproc)"
echo ""

# =======================================================================
# CONFIGURATION
# =======================================================================

# Search type: 'grid' for exhaustive search, 'random' for sampling
SEARCH_TYPE=${SEARCH_TYPE:-random}  # Default to random (faster)
N_RANDOM=${N_RANDOM:-20}            # Number of random combinations

echo "=== SEARCH CONFIGURATION ==="
echo "Search type: $SEARCH_TYPE"
echo "Image size: 512×512 (original resolution)"
echo "Batch sizes: [4, 8, 16] (reduced for higher resolution)"
if [ "$SEARCH_TYPE" = "random" ]; then
    echo "Random combinations: $N_RANDOM"
    echo "Estimated time: $(($N_RANDOM * 8))-$(($N_RANDOM * 15)) minutes"
else
    echo "Grid search: 108 combinations (full factorial)"
    echo "Estimated time: 12-24 hours"
fi
echo "============================"
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
else
    echo "   ERROR: Dataset not found!"
    echo "   Expected: ./dataset_microscope/images/ and ./dataset_microscope/masks/"
    exit 1
fi

# Check 2: Search script
echo ""
echo "2. Checking hyperparameter search script..."
if [ -f "./hyperparameter_search_microbead.py" ]; then
    echo "   ✓ Search script found"
else
    echo "   ERROR: hyperparameter_search_microbead.py not found"
    echo "   Please ensure the file is uploaded to HPC"
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

echo "============================="
echo ""

# =======================================================================
# TENSORFLOW AND GPU STATUS CHECK
# =======================================================================

echo "=== TENSORFLOW & GPU STATUS ===" singularity exec --nv "$image" python3 -c "
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
    print('WARNING: No GPUs detected - search will be VERY SLOW on CPU')
"
echo "==============================="
echo ""

# =======================================================================
# RUN HYPERPARAMETER SEARCH
# =======================================================================

echo "🔍 STARTING HYPERPARAMETER SEARCH"
echo "=============================================="
echo ""

if [ "$SEARCH_TYPE" = "grid" ]; then
    echo "Running FULL GRID SEARCH (108 combinations)"
    echo "This will test all combinations of:"
    echo "  - Image size: 512×512 (original resolution)"
    echo "  - Learning rates: 5e-5, 1e-4, 2e-4"
    echo "  - Batch sizes: 4, 8, 16 (reduced for 512×512)"
    echo "  - Dropout rates: 0.0, 0.1, 0.2, 0.3"
    echo "  - Loss functions: dice, focal, combined"
    echo ""
    echo "⏱️ Estimated completion: 12-24 hours"
else
    echo "Running RANDOM SEARCH ($N_RANDOM combinations)"
    echo "Randomly sampling from:"
    echo "  - Image size: 512×512 (original resolution)"
    echo "  - Learning rates: 5e-5, 1e-4, 2e-4"
    echo "  - Batch sizes: 4, 8, 16 (reduced for 512×512)"
    echo "  - Dropout rates: 0.0, 0.1, 0.2, 0.3"
    echo "  - Loss functions: dice, focal, combined"
    echo ""
    echo "⏱️ Estimated completion: $(($N_RANDOM * 8))-$(($N_RANDOM * 15)) minutes"
fi

echo "=============================================="
echo ""

# Execute search
if [ "$SEARCH_TYPE" = "grid" ]; then
    singularity exec --nv "$image" python3 hyperparameter_search_microbead.py --search-type grid
else
    singularity exec --nv "$image" python3 hyperparameter_search_microbead.py --search-type random --n-random $N_RANDOM
fi

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
    echo "✓ Search completed successfully!"
    echo ""

    # Find output directory
    SEARCH_OUTPUT=$(ls -td hyperparam_search_* 2>/dev/null | head -1)

    if [ -n "$SEARCH_OUTPUT" ] && [ -d "$SEARCH_OUTPUT" ]; then
        echo "📊 SEARCH RESULTS:"
        echo "=================="

        # Display top 5 results
        if [ -f "$SEARCH_OUTPUT/search_results_final.csv" ]; then
            echo ""
            echo "Top 5 hyperparameter configurations:"
            echo ""
            head -6 "$SEARCH_OUTPUT/search_results_final.csv"
            echo ""
        fi

        # Display best configuration
        if [ -f "$SEARCH_OUTPUT/best_hyperparameters.json" ]; then
            echo "🏆 BEST CONFIGURATION:"
            echo "====================="
            cat "$SEARCH_OUTPUT/best_hyperparameters.json"
            echo ""
        fi

        # Count files
        model_count=$(find "$SEARCH_OUTPUT" -name "*.hdf5" | wc -l)
        csv_count=$(find "$SEARCH_OUTPUT" -name "*.csv" | wc -l)

        echo "Generated files:"
        echo "  - Model files (.hdf5): $model_count"
        echo "  - History files (.csv): $csv_count"
        echo "  - Results summary: search_results_final.csv"
        echo "  - Best config: best_hyperparameters.json"
        echo ""

        echo "📁 OUTPUT DIRECTORY: $SEARCH_OUTPUT"
        echo ""

        echo "📥 TO DOWNLOAD RESULTS:"
        echo "======================="
        echo "From your local machine:"
        echo "  scp -r phyzxi@hpc:~/scratch/unet-HPC/$SEARCH_OUTPUT ./hyperparam_results/"
        echo ""

        # Analyze results
        singularity exec --nv "$image" python3 -c "
import pandas as pd
import os

output_dir = '$SEARCH_OUTPUT'
results_file = os.path.join(output_dir, 'search_results_final.csv')

if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    best = df.iloc[0]

    print('📈 PERFORMANCE ANALYSIS:')
    print('========================')
    print(f'Best Val Jaccard: {best[\"best_val_jacard\"]:.4f}')
    print(f'Improvement over baseline (0.2456): {best[\"best_val_jacard\"]/0.2456:.2f}×')
    print()

    # Previous best from microbead_training_20251009_073134
    baseline = 0.2456

    if best['best_val_jacard'] >= 0.50:
        print('✅ EXCELLENT: Reached target performance (≥0.50)!')
        print('   → Production-ready segmentation')
    elif best['best_val_jacard'] >= 0.35:
        print('✓✓ VERY GOOD: Significant improvement!')
        print(f'   → {best[\"best_val_jacard\"]/baseline:.2f}× better than baseline')
    elif best['best_val_jacard'] > baseline:
        print('✓ GOOD: Performance improved')
        print(f'   → {best[\"best_val_jacard\"]/baseline:.2f}× better than baseline')
    else:
        print('⚠ WARNING: No improvement over baseline')
        print('   → Further investigation needed')

    print()
    print('🔍 KEY INSIGHTS:')
    print('================')

    # Analyze hyperparameter trends
    if 'dropout' in df.columns:
        best_dropout_configs = df.nsmallest(5, 'best_val_jacard')
        avg_dropout = df.groupby('dropout')['best_val_jacard'].mean().sort_values(ascending=False)
        print(f'Best dropout values: {avg_dropout.head(2).to_dict()}')

    if 'loss_type' in df.columns:
        avg_loss = df.groupby('loss_type')['best_val_jacard'].mean().sort_values(ascending=False)
        print(f'Best loss types: {avg_loss.to_dict()}')

    if 'learning_rate' in df.columns:
        avg_lr = df.groupby('learning_rate')['best_val_jacard'].mean().sort_values(ascending=False)
        print(f'Best learning rates: {avg_lr.to_dict()}')
else:
    print('Results file not found')
" 2>/dev/null || echo "Could not analyze results"

    fi

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Search failed!"
    echo ""

    echo "🔍 ERROR ANALYSIS:"
    echo "=================="
    echo "Check the job output above for error messages"
    echo ""

    echo "🔧 COMMON ISSUES:"
    echo "================="
    echo "1. Out of memory: Reduce batch size or number of parallel experiments"
    echo "2. Dataset path: Verify dataset_microscope/images/ and masks/ exist"
    echo "3. Model file: Ensure models.py is present"
    echo "4. Dependencies: Check all packages installed in container"
fi

echo ""
echo "📝 FULL LOGS:"
echo "============="
echo "  PBS output: Microbead_Hyperparam_Search.o${PBS_JOBID##*.}"
echo ""
echo "======================================================================="
echo "HYPERPARAMETER SEARCH - JOB COMPLETE"
echo "======================================================================="
echo "Search type: $SEARCH_TYPE"
if [ "$SEARCH_TYPE" = "random" ]; then
    echo "Combinations tested: $N_RANDOM"
else
    echo "Combinations tested: 108 (full grid)"
fi
echo "======================================================================="
