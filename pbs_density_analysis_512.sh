#!/bin/bash
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Density_512
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# DENSITY ANALYSIS USING BEST 512×512 MODEL
# =======================================================================
# Uses best configuration from hyperparameter_search_512 results
# Trains final model on full dataset
# Performs density analysis on ./test_images/ with dilution series
# =======================================================================

cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "DENSITY ANALYSIS WITH BEST 512×512 MODEL"
echo "======================================================================="
echo "Framework: TensorFlow/Keras with Mixed Precision"
echo ""

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo ""

# =======================================================================
# CONFIGURATION - UPDATE THIS WITH YOUR HYPERPARAMETER SEARCH RESULTS!
# =======================================================================

echo "=== CONFIGURATION ==="
echo ""
echo "⚠ IMPORTANT: Update HYPERPARAM_SEARCH_DIR with your results directory!"
echo ""

# TODO: UPDATE THIS LINE WITH YOUR HYPERPARAMETER SEARCH OUTPUT DIRECTORY!
HYPERPARAM_SEARCH_DIR="./hyperparameter_search_512_YYYYMMDD_HHMMSS"

echo "Hyperparameter search directory:"
echo "  $HYPERPARAM_SEARCH_DIR"
echo ""

# Check if directory exists
if [ ! -d "$HYPERPARAM_SEARCH_DIR" ]; then
    echo "✗ ERROR: Hyperparameter search directory not found!"
    echo ""
    echo "Please update HYPERPARAM_SEARCH_DIR in this PBS script."
    echo "Example: HYPERPARAM_SEARCH_DIR=\"./hyperparameter_search_512_20251014_123456\""
    echo ""
    echo "Available directories:"
    find . -maxdepth 1 -type d -name "hyperparameter_search_512_*" | sort
    echo ""
    exit 1
fi

# Check if summary exists
if [ ! -f "$HYPERPARAM_SEARCH_DIR/summary.json" ]; then
    echo "✗ ERROR: summary.json not found in $HYPERPARAM_SEARCH_DIR"
    echo "Make sure hyperparameter search completed successfully."
    exit 1
fi

echo "✓ Found hyperparameter search results"
echo ""

# Display best configuration
echo "Best Configuration from Hyperparameter Search:"
echo "----------------------------------------------"
python3 -c "
import json
with open('$HYPERPARAM_SEARCH_DIR/summary.json', 'r') as f:
    summary = json.load(f)
print(f\"Config: {summary['best_config']}\")
print(f\"CV Jaccard: {summary['best_jaccard']:.4f}\")
print(f\"Successful configs: {summary['successful_configs']}/{summary['total_configs']}\")
"
echo "----------------------------------------------"
echo ""

echo "Workflow:"
echo "  Phase 1: Train final model on full dataset (90/10 split)"
echo "  Phase 2: Predict on test images (./test_images/)"
echo "  Phase 3: Generate density analysis (boxplots + tile comparisons)"
echo ""
echo "Expected Runtime: 2-4 hours"
echo "====================="
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
    echo "ERROR: TensorFlow container not found"
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
if [ ! -f "./density_analysis_512_best_model.py" ]; then
    echo "   ✗ ERROR: density_analysis_512_best_model.py not found!"
    exit 1
fi
echo "   ✓ density_analysis_512_best_model.py found"

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
echo "2. Checking training dataset..."
if [ ! -d "./dataset_shrunk_masks/images/" ] || [ ! -d "./dataset_shrunk_masks/masks/" ]; then
    echo "   ✗ ERROR: dataset_shrunk_masks not found!"
    exit 1
fi

img_count=$(find ./dataset_shrunk_masks/images/ -name "*.png" 2>/dev/null | wc -l)
echo "   ✓ Training images: $img_count .png files"

echo ""
echo "3. Checking test images..."
if [ ! -d "./test_images/" ]; then
    echo "   ✗ ERROR: Test images directory not found!"
    exit 1
fi

test_img_count=$(find ./test_images/ -name "*.tif" 2>/dev/null | wc -l)
echo "   ✓ Test images: $test_img_count .tif files"

if [ $test_img_count -eq 0 ]; then
    echo "   ✗ ERROR: No test images found!"
    exit 1
fi

echo "   Sample test images:"
find ./test_images/ -name "*.tif" 2>/dev/null | head -5 | while read img; do
    echo "     - $(basename $img)"
done

echo "=============================="
echo ""

# =======================================================================
# TENSORFLOW & GPU CHECK
# =======================================================================

echo "=== TENSORFLOW & GPU STATUS ==="
singularity exec --nv "$image" python3 -c "
import tensorflow as tf

print('TensorFlow version:', tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
print('Physical GPUs found:', len(gpus))

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('✓ GPU memory growth enabled')

    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0]])
        b = tf.constant([[2.0], [0.0]])
        c = tf.matmul(a, b)
    print('✓ GPU operation successful')
else:
    print('⚠ No GPU detected')
"

echo "================================"
echo ""

# =======================================================================
# EXECUTE DENSITY ANALYSIS
# =======================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🚀 STARTING DENSITY ANALYSIS"
echo "=============================================="
echo "Phase 1: Train final model (~1-2 hours)"
echo "  - Uses best config from hyperparameter search"
echo "  - Trains on full dataset (90/10 split)"
echo "  - Saves best model"
echo ""
echo "Phase 2: Predict on test images (~30 min)"
echo "  - Processes ./test_images/ with 512×512 tiles"
echo "  - Calculates foreground percentage"
echo "  - Selects 5 representative tiles per image"
echo ""
echo "Phase 3: Generate outputs (~10 min)"
echo "  - Creates boxplots (DL model + CLAHE+OTSU)"
echo "  - Generates tile comparison images"
echo "  - Exports comprehensive CSV"
echo ""
echo "Progress logged below..."
echo "=============================================="
echo ""

singularity exec --nv "$image" python3 density_analysis_512_best_model.py "$HYPERPARAM_SEARCH_DIR" 2>&1 | tee "density_analysis_512_console_${TIMESTAMP}.log"

EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "DENSITY ANALYSIS COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✓ SUCCESS"
    echo ""
    echo "✓ Density analysis completed successfully!"
    echo ""

    # Find output directory
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "density_analysis_512_best_*" -mtime -1 | sort | tail -1)

    if [ -d "$OUTPUT_DIR" ]; then
        echo "📁 RESULTS DIRECTORY: $OUTPUT_DIR"
        echo ""

        # Count generated files
        model_count=$(find "$OUTPUT_DIR/trained_model" -name "*.keras" 2>/dev/null | wc -l)
        tile_count=$(find "$OUTPUT_DIR/representative_tiles" -name "*.png" 2>/dev/null | wc -l)
        plot_count=$(find "$OUTPUT_DIR/boxplots" -name "*.png" 2>/dev/null | wc -l)
        csv_count=$(find "$OUTPUT_DIR/csv_data" -name "*.csv" 2>/dev/null | wc -l)

        echo "📊 GENERATED FILES:"
        echo "  Final model: $model_count .keras file"
        echo "  Representative tile comparisons: $tile_count"
        echo "  Boxplots: $plot_count"
        echo "  CSV files: $csv_count"
        echo ""

        if [ $tile_count -gt 0 ]; then
            echo "🖼️  REPRESENTATIVE TILES GENERATED:"
            echo "   (2-panel: Original 512×512 | Predicted mask)"
            find "$OUTPUT_DIR/representative_tiles" -name "*.png" | head -10 | while read img; do
                echo "   - $(basename $img)"
            done
            if [ $tile_count -gt 10 ]; then
                echo "   ... and $((tile_count - 10)) more"
            fi
            echo ""
        fi

        if [ $plot_count -gt 0 ]; then
            echo "📈 BOXPLOTS GENERATED:"
            find "$OUTPUT_DIR/boxplots" -name "*.png" -exec basename {} \; | sort
            echo ""
        fi

        if [ -f "$OUTPUT_DIR/csv_data/density_analysis_comprehensive.csv" ]; then
            echo "📄 COMPREHENSIVE CSV:"
            echo "   $OUTPUT_DIR/csv_data/density_analysis_comprehensive.csv"
            echo ""

            singularity exec --nv "$image" python3 -c "
import pandas as pd
try:
    df = pd.read_csv('$OUTPUT_DIR/csv_data/density_analysis_comprehensive.csv')
    print('CSV SUMMARY:')
    print('-' * 60)
    print(f'Total measurements: {len(df)}')
    print(f'Methods: {df[\"method\"].unique().tolist()}')
    print(f'Dilution factors: {sorted(df[\"dilution_factor\"].unique())}')
    print(f'Images analyzed: {df[\"image\"].nunique()}')
    print()
    print('Mean Foreground Percentage by Method:')
    for method in df['method'].unique():
        mean_fg = df[df['method'] == method]['foreground_pct'].mean()
        print(f'  {method.ljust(20)}: {mean_fg:.4f}%')
except Exception as e:
    print(f'Unable to parse CSV: {e}')
" 2>/dev/null || echo "   Results saved."
        fi

    else
        echo "⚠ WARNING: Expected output directory not found!"
    fi

    echo ""
    echo "🎯 NEXT STEPS:"
    echo "============="
    echo "1. Review boxplots:"
    echo "   Open $OUTPUT_DIR/boxplots/"
    echo ""
    echo "2. Examine tile comparisons:"
    echo "   Check $OUTPUT_DIR/representative_tiles/"
    echo ""
    echo "3. Analyze CSV data:"
    echo "   $OUTPUT_DIR/csv_data/density_analysis_comprehensive.csv"
    echo ""
    echo "4. Compare with 256×256 results:"
    echo "   Does 512×512 provide better segmentation?"
    echo "   Check if density trends match across resolutions"

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Density analysis failed!"
    echo ""

    if [ -f "density_analysis_512_console_${TIMESTAMP}.log" ]; then
        echo "Last 30 lines of console log:"
        echo "-----------------------------"
        tail -30 "density_analysis_512_console_${TIMESTAMP}.log"
    fi

    echo ""
    echo "🔧 COMMON ISSUES:"
    echo "================="
    echo "1. Hyperparameter search directory not found"
    echo "   - Update HYPERPARAM_SEARCH_DIR in this script"
    echo ""
    echo "2. Training dataset missing"
    echo "   - Check dataset_shrunk_masks/images/ and masks/"
    echo ""
    echo "3. OOM during training"
    echo "   - Check if batch size from best config is too large"
    echo "   - May need to reduce batch size manually"
fi

echo ""
echo "📝 CONSOLE LOG: density_analysis_512_console_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "DENSITY ANALYSIS JOB COMPLETE"
echo "Tile size: 512×512"
echo "Best model from hyperparameter search"
echo "======================================="
