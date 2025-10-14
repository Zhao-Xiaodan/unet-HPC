#!/bin/bash
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Density_Pred
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# DENSITY PREDICTION WITH REPRESENTATIVE TILE VISUALIZATION
# =======================================================================
# Trains models using validation_arch_comparison configuration,
# then predicts on test images with 512×512 tiles.
#
# Output:
# - 5 representative tile comparisons per test image (original + 3 masks)
# - 4 boxplots (one per architecture/method)
# - 1 comprehensive CSV with all density data
# =======================================================================

cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "DENSITY PREDICTION WITH REPRESENTATIVE TILES"
echo "======================================================================="
echo "Framework: TensorFlow/Keras"
echo ""

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# =======================================================================
# CONFIGURATION
# =======================================================================

echo "=== CONFIGURATION ==="
echo "Phase 1: Train models (U-Net, ResUNet, Attention ResUNet)"
echo "  Config: validation_arch_comparison settings"
echo "  Filters: 64, Dropout: 0.2, Batch: 16, LR: 5e-5"
echo "  Training size: 256×256"
echo ""
echo "Phase 2: Predict on test images"
echo "  Tile size: 512×512 (as requested)"
echo "  Test images: ./test_images/ (dilution series)"
echo ""
echo "Phase 3: Generate outputs"
echo "  - 5 representative tiles per image"
echo "    (min, 25th percentile, median, 75th percentile, max density)"
echo "  - 4-panel comparison: original + 3 predicted masks"
echo "  - 4 boxplots (one per method)"
echo "  - 1 comprehensive CSV"
echo ""
echo "Expected Runtime: 4-6 hours"
echo "====================="
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
if [ ! -f "./density_prediction_with_tiles.py" ]; then
    echo "   ✗ ERROR: density_prediction_with_tiles.py not found!"
    exit 1
fi
echo "   ✓ density_prediction_with_tiles.py found"

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
if [ ! -d "./dataset_full_stack/images/" ] || [ ! -d "./dataset_full_stack/masks/" ]; then
    echo "   ✗ ERROR: Training dataset not found!"
    exit 1
fi

train_img_count=$(find ./dataset_full_stack/images/ -name "*.tif" 2>/dev/null | wc -l)
echo "   ✓ Training images: $train_img_count .tif files"

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
import sys

print('Python version:', sys.version.split()[0])
print('TensorFlow version:', tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
print('Physical GPUs found:', len(gpus))

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('✓ GPU memory growth enabled')

    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[2.0, 0.0], [0.0, 2.0]])
        c = tf.matmul(a, b)
    print('✓ Basic GPU operation successful')
else:
    print('⚠ WARNING: No GPU detected!')

print()
print('Checking dependencies:')
for dep in ['cv2', 'PIL', 'matplotlib', 'numpy', 'sklearn', 'pandas', 'seaborn', 'tqdm']:
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
# EXECUTE DENSITY PREDICTION
# =======================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🚀 STARTING DENSITY PREDICTION"
echo "=============================================="
echo "Phase 1: Train 3 models (~3-4 hours)"
echo "Phase 2: Predict on test images (~1-2 hours)"
echo "Phase 3: Generate visualizations and CSV"
echo ""
echo "Progress logged below..."
echo "=============================================="
echo ""

singularity exec --nv "$image" python3 density_prediction_with_tiles.py 2>&1 | tee "density_prediction_console_${TIMESTAMP}.log"

EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "DENSITY PREDICTION COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✓ SUCCESS"
    echo ""
    echo "✓ Density prediction completed successfully!"
    echo ""

    # Find output directory
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "density_prediction_*" -mtime -1 | sort | tail -1)

    if [ -d "$OUTPUT_DIR" ]; then
        echo "📁 RESULTS DIRECTORY: $OUTPUT_DIR"
        echo ""

        # Count files
        model_count=$(find "$OUTPUT_DIR/trained_models" -name "*.keras" 2>/dev/null | wc -l)
        tile_count=$(find "$OUTPUT_DIR/representative_tiles" -name "*.png" 2>/dev/null | wc -l)
        plot_count=$(find "$OUTPUT_DIR/boxplots" -name "*.png" 2>/dev/null | wc -l)
        csv_count=$(find "$OUTPUT_DIR/csv_data" -name "*.csv" 2>/dev/null | wc -l)

        echo "📊 GENERATED FILES:"
        echo "  Models trained: $model_count"
        echo "  Representative tile comparisons: $tile_count"
        echo "  Boxplots: $plot_count"
        echo "  CSV files: $csv_count"
        echo ""

        if [ $tile_count -gt 0 ]; then
            echo "🖼️  REPRESENTATIVE TILES GENERATED:"
            echo "   (Each shows: Original | U-Net | ResUNet | Attention ResUNet)"
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
    echo "1. Review representative tile comparisons:"
    echo "   Look for tiles showing original + 3 predicted masks side-by-side"
    echo ""
    echo "2. Compare architectures visually:"
    echo "   Check which architecture best matches ground truth patterns"
    echo ""
    echo "3. Analyze density trends:"
    echo "   Review boxplots for density vs dilution factor relationships"
    echo ""
    echo "4. Examine comprehensive CSV:"
    echo "   Statistical analysis of performance across dilution series"

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Density prediction failed!"
    echo ""

    if [ -f "density_prediction_console_${TIMESTAMP}.log" ]; then
        echo "Last 30 lines of console log:"
        echo "-----------------------------"
        tail -30 "density_prediction_console_${TIMESTAMP}.log"
    fi

    echo ""
    echo "🔧 COMMON ISSUES:"
    echo "================="
    echo "1. Training dataset missing"
    echo "2. Test images missing or incorrect format"
    echo "3. GPU memory issues (try reducing batch_size)"
    echo "4. Model architecture import errors"
fi

echo ""
echo "📝 CONSOLE LOG: density_prediction_console_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "DENSITY PREDICTION JOB COMPLETE"
echo "Tile size: 512×512"
echo "Architectures: U-Net, ResUNet, Attention ResUNet"
echo "======================================="
