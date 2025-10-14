#!/bin/bash
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Density_Exist
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# FAST DENSITY PREDICTION - EXISTING MODELS ONLY
# =======================================================================
# Uses trained models from validation_arch_comparison_20251013_093844
# NO TRAINING - Only loads models and predicts
# Expected runtime: ~5 minutes
# =======================================================================

cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "FAST DENSITY PREDICTION - EXISTING MODELS"
echo "======================================================================="
echo "Loading pre-trained models from validation_arch_comparison"
echo "NO TRAINING - Prediction only"
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
echo "Model source: validation_arch_comparison_20251013_093844"
echo ""
echo "Execution mode: LOAD EXISTING MODELS"
echo "  Expected runtime: ~5 minutes"
echo "  - Search and load models (~30 seconds)"
echo "  - Predict on test images (~3-4 minutes)"
echo "  - Generate visualizations and CSV (~1 minute)"
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
echo "============================"
echo ""

# =======================================================================
# PRE-EXECUTION CHECKS
# =======================================================================

echo "=== PRE-EXECUTION CHECKS ==="

echo "1. Checking Python scripts..."
if [ ! -f "./density_prediction_existing_models.py" ]; then
    echo "   ✗ ERROR: density_prediction_existing_models.py not found!"
    exit 1
fi
echo "   ✓ density_prediction_existing_models.py found"

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
echo "2. Checking for validation_arch_comparison directory..."
if [ ! -d "./validation_arch_comparison_20251013_093844" ]; then
    echo "   ✗ ERROR: validation_arch_comparison_20251013_093844 directory not found!"
    echo "   This directory should contain trained models from architecture comparison."
    exit 1
fi
echo "   ✓ validation_arch_comparison_20251013_093844 found"

# Check for architecture subdirectories
echo ""
echo "3. Checking for architecture subdirectories..."
ARCH_FOUND=0
for arch in unet resunet attention_resunet; do
    if [ -d "./validation_arch_comparison_20251013_093844/$arch" ]; then
        fold_count=$(find "./validation_arch_comparison_20251013_093844/$arch" -maxdepth 1 -type d -name "fold_*" | wc -l)
        echo "   ✓ $arch: $fold_count fold(s) found"
        ARCH_FOUND=$((ARCH_FOUND + 1))
    else
        echo "   ⚠ $arch: directory not found"
    fi
done

if [ $ARCH_FOUND -eq 0 ]; then
    echo ""
    echo "   ✗ ERROR: No architecture directories found!"
    echo "   Expected directories:"
    echo "     - validation_arch_comparison_20251013_093844/unet/"
    echo "     - validation_arch_comparison_20251013_093844/resunet/"
    echo "     - validation_arch_comparison_20251013_093844/attention_resunet/"
    exit 1
fi

echo ""
echo "4. Checking for model files in fold directories..."
MODEL_FOUND=0
for arch in unet resunet attention_resunet; do
    arch_dir="./validation_arch_comparison_20251013_093844/$arch"
    if [ -d "$arch_dir" ]; then
        for fold_dir in "$arch_dir"/fold_*; do
            if [ -d "$fold_dir" ]; then
                # Check for various model file formats
                model_count=$(find "$fold_dir" -maxdepth 1 -type f \( -name "*.keras" -o -name "*.h5" -o -name "*.hdf5" \) | wc -l)
                if [ $model_count -gt 0 ]; then
                    echo "   ✓ $arch/$(basename $fold_dir): $model_count model file(s)"
                    MODEL_FOUND=$((MODEL_FOUND + 1))
                fi

                # Check for results.json (to find best fold)
                if [ -f "$fold_dir/results.json" ]; then
                    echo "      - results.json found (for best fold selection)"
                fi
            fi
        done
    fi
done

if [ $MODEL_FOUND -eq 0 ]; then
    echo ""
    echo "   ⚠ WARNING: No model files (.keras, .h5, .hdf5) found in fold directories!"
    echo "   Script will attempt to search for models, but may fail if not found."
    echo ""
else
    echo ""
    echo "   ✓ Found model files in $MODEL_FOUND fold(s)"
fi

echo ""
echo "5. Checking test images..."
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

echo "==============================="
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

echo "🚀 STARTING FAST DENSITY PREDICTION"
echo "=============================================="
echo "Mode: LOAD EXISTING MODELS (no training)"
echo "Expected runtime: ~5 minutes"
echo "  - Search for models in validation_arch_comparison (~5 seconds)"
echo "  - Identify best fold per architecture (~5 seconds)"
echo "  - Load 3 models from disk (~20 seconds)"
echo "  - Predict on test images (~3-4 minutes)"
echo "  - Generate visualizations and CSV (~1 minute)"
echo ""
echo "Progress logged below..."
echo "=============================================="
echo ""

singularity exec --nv "$image" python3 density_prediction_existing_models.py 2>&1 | tee "density_existing_console_${TIMESTAMP}.log"

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
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "density_prediction_existing_*" -mtime -1 | sort | tail -1)

    if [ -d "$OUTPUT_DIR" ]; then
        echo "📁 RESULTS DIRECTORY: $OUTPUT_DIR"
        echo ""

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
    echo "   - Check statistical robustness (256×256 has 4× more tiles)"
    echo ""
    echo "💡 To re-run: Just submit again - will be fast (~5 min)!"

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Density prediction failed!"
    echo ""

    if [ -f "density_existing_console_${TIMESTAMP}.log" ]; then
        echo "Last 30 lines of console log:"
        echo "-----------------------------"
        tail -30 "density_existing_console_${TIMESTAMP}.log"
    fi

    echo ""
    echo "🔧 COMMON ISSUES:"
    echo "================="
    echo "1. Model files not found in validation_arch_comparison_20251013_093844"
    echo "   - Check if fold directories contain .keras, .h5, or .hdf5 files"
    echo "   - Verify results.json files exist for best fold selection"
    echo ""
    echo "2. Missing custom loss/metrics modules"
    echo "   - Ensure loss_functions_fixed.py is present"
    echo ""
    echo "3. Test images missing or wrong format"
    echo "   - Check ./test_images/ directory for .tif files"
    echo ""
    echo "4. GPU memory issues"
    echo "   - Reduce batch_size in script if needed"
fi

echo ""
echo "📝 CONSOLE LOG: density_existing_console_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "DENSITY PREDICTION JOB COMPLETE"
echo "Tile size: 256×256 (native resolution)"
echo "Models: Pre-trained (no training phase)"
echo "======================================="
