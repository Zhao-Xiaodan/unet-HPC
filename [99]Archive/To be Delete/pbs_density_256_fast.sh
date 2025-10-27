#!/bin/bash
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Density_256
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# FAST DENSITY PREDICTION - 256×256 TILES
# =======================================================================
# Smart model management:
# - First run: Trains models once, saves them (~3-4 hours)
# - Subsequent runs: Loads saved models (~5 minutes)
#
# Features:
# - 256×256 tiles (native model resolution - no interpolation)
# - Representative tile visualization (5 per image)
# - Density analysis with boxplots
# =======================================================================

cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "FAST DENSITY PREDICTION - 256×256 TILES"
echo "======================================================================="
echo "Smart model management for fast execution"
echo ""

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo ""

# =======================================================================
# CONFIGURATION
# =======================================================================

echo "=== CONFIGURATION ==="
echo "Tile size: 256×256 (native model resolution)"
echo "Architectures: U-Net, ResUNet, Attention ResUNet"
echo ""
echo "Execution modes:"
echo "  First run: Train models once, save to ./saved_models_validation_config/"
echo "             Runtime: ~3-4 hours"
echo "  Subsequent runs: Load saved models from disk"
echo "                   Runtime: ~5 minutes"
echo ""
echo "Output:"
echo "  - 5 representative tile comparisons per test image"
echo "  - 4 boxplots (one per method)"
echo "  - 1 comprehensive CSV"
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
    echo "ERROR: TensorFlow container not found"
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
if [ ! -f "./density_prediction_256_fast.py" ]; then
    echo "   ✗ ERROR: density_prediction_256_fast.py not found!"
    exit 1
fi
echo "   ✓ density_prediction_256_fast.py found"

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
echo "2. Checking for existing models..."
if [ -d "./saved_models_validation_config" ]; then
    model_count=$(find ./saved_models_validation_config -name "*.keras" 2>/dev/null | wc -l)
    echo "   ✓ Models directory exists"
    echo "   ✓ Found $model_count saved model(s)"

    if [ $model_count -eq 3 ]; then
        echo "   ✅ All 3 models found - FAST MODE (~5 min execution)"
        FAST_MODE=true
    else
        echo "   ⚠ Only $model_count/3 models found - will train missing models"
        FAST_MODE=false
    fi
else
    echo "   ⚠ Models directory doesn't exist - will train all models"
    FAST_MODE=false
fi

echo ""
echo "3. Checking training dataset..."
if [ ! -d "./dataset_full_stack/images/" ] || [ ! -d "./dataset_full_stack/masks/" ]; then
    if [ "$FAST_MODE" = false ]; then
        echo "   ✗ ERROR: Training dataset required (models not found)"
        exit 1
    else
        echo "   ⚠ Training dataset not checked (not needed for fast mode)"
    fi
else
    train_img_count=$(find ./dataset_full_stack/images/ -name "*.tif" 2>/dev/null | wc -l)
    echo "   ✓ Training images: $train_img_count .tif files"
fi

echo ""
echo "4. Checking test images..."
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
    print('✓ GPU ready')
else:
    print('⚠ No GPU')
"

echo "================================"
echo ""

# =======================================================================
# EXECUTE DENSITY PREDICTION
# =======================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ "$FAST_MODE" = true ]; then
    echo "🚀 STARTING FAST DENSITY PREDICTION"
    echo "=============================================="
    echo "Mode: FAST (models already trained)"
    echo "Expected runtime: ~5 minutes"
    echo "  - Load 3 models from disk (~30 seconds)"
    echo "  - Predict on test images (~3-4 minutes)"
    echo "  - Generate visualizations and CSV (~1 minute)"
else
    echo "🚀 STARTING DENSITY PREDICTION"
    echo "=============================================="
    echo "Mode: TRAIN + PREDICT (first run)"
    echo "Expected runtime: ~3-4 hours"
    echo "  - Train missing models (~3-4 hours)"
    echo "  - Save models to ./saved_models_validation_config/"
    echo "  - Predict on test images (~3-4 minutes)"
    echo "  - Generate visualizations and CSV (~1 minute)"
    echo ""
    echo "💡 Future runs will be FAST (~5 min) after models are saved!"
fi
echo ""
echo "Progress logged below..."
echo "=============================================="
echo ""

singularity exec --nv "$image" python3 density_prediction_256_fast.py 2>&1 | tee "density_256_console_${TIMESTAMP}.log"

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
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "density_prediction_256_*" -mtime -1 | sort | tail -1)

    if [ -d "$OUTPUT_DIR" ]; then
        echo "📁 RESULTS DIRECTORY: $OUTPUT_DIR"
        echo ""

        # Check if models were saved
        if [ -d "./saved_models_validation_config" ]; then
            model_count=$(find ./saved_models_validation_config -name "*.keras" 2>/dev/null | wc -l)
            echo "💾 SAVED MODELS: $model_count model(s) in ./saved_models_validation_config/"
            if [ $model_count -eq 3 ]; then
                echo "   ✅ All 3 models saved - next run will be FAST (~5 min)!"
            fi
            echo ""
        fi

        # Count generated files
        tile_count=$(find "$OUTPUT_DIR/representative_tiles" -name "*.png" 2>/dev/null | wc -l)
        plot_count=$(find "$OUTPUT_DIR/boxplots" -name "*.png" 2>/dev/null | wc -l)
        csv_count=$(find "$OUTPUT_DIR/csv_data" -name "*.csv" 2>/dev/null | wc -l)

        echo "📊 GENERATED FILES:"
        echo "  Representative tile comparisons: $tile_count"
        echo "  Boxplots: $plot_count"
        echo "  CSV files: $csv_count"
        echo ""

        if [ $tile_count -gt 0 ]; then
            echo "🖼️  REPRESENTATIVE TILES (256×256):"
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
    print(f'Tiles per image: {len(df) // (df[\"image\"].nunique() * len(df[\"method\"].unique()))}')
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
    echo "1. Review representative tile comparisons (256×256)"
    echo "   - Compare visual quality across architectures"
    echo "   - Check if all architectures detect small particles"
    echo ""
    echo "2. Analyze density trends in boxplots"
    echo "   - Verify expected density decrease with dilution"
    echo ""
    echo "3. Compare with 512×512 results (from other job)"
    echo "   - Which tile size gives better segmentation?"
    echo "   - Check if 256×256 captures all particle details"
    echo ""
    if [ "$FAST_MODE" = true ]; then
        echo "💡 Want to re-run prediction? Just submit again - will be fast (~5 min)!"
    else
        echo "💡 Models saved! Future runs will be fast (~5 min)!"
    fi

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Density prediction failed!"
    echo ""

    if [ -f "density_256_console_${TIMESTAMP}.log" ]; then
        echo "Last 30 lines of console log:"
        echo "-----------------------------"
        tail -30 "density_256_console_${TIMESTAMP}.log"
    fi

    echo ""
    echo "🔧 COMMON ISSUES:"
    echo "================="
    echo "1. First run: Needs training dataset (for model training)"
    echo "2. GPU memory issues (reduce batch_size in script)"
    echo "3. Test images missing or wrong format"
fi

echo ""
echo "📝 CONSOLE LOG: density_256_console_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "DENSITY PREDICTION JOB COMPLETE"
echo "Tile size: 256×256 (native resolution)"
echo "======================================="
