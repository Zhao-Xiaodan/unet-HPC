#!/bin/bash
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Density_512_Gray
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# DENSITY ANALYSIS - 512×512 GRAYSCALE MODELS
# =======================================================================
# Uses top 5 models from hyperparameter_search_512_20251014_235755
# Predicts on test_images/ and generates density analysis
#
# Output:
# - 5 individual boxplots (one per model)
# - 4-panel tile comparisons (Original + top 3 U-Net models)
# - Comprehensive CSV with all results
# =======================================================================

cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "DENSITY ANALYSIS - 512×512 GRAYSCALE MODELS"
echo "======================================================================="
echo "Script: density_analysis_512_grayscale.py"
echo "PBS Script: pbs_density_analysis_512.sh"
echo ""
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $(free -h | grep Mem | awk '{print $2}'), CPUs: $(nproc)"
echo ""

# =======================================================================
# MODEL CONFIGURATION
# =======================================================================

echo "=== MODELS TO BE USED ==="
echo "Source: hyperparameter_search_512_20251014_235755/"
echo ""
echo "Top 5 configurations (by 3-fold CV performance):"
echo "  1. unet_lr0.0001_drop0.3_bs4"
echo "     Mean Jaccard: 0.1533 ± 0.0578 (BEST)"
echo "     Architecture: U-Net"
echo ""
echo "  2. unet_lr5e-05_drop0.2_bs4"
echo "     Mean Jaccard: 0.1327 ± 0.0176"
echo "     Architecture: U-Net"
echo ""
echo "  3. unet_lr5e-05_drop0.3_bs4"
echo "     Mean Jaccard: 0.1308 ± 0.0137"
echo "     Architecture: U-Net"
echo ""
echo "  4. resunet_lr5e-05_drop0.3_bs4"
echo "     Mean Jaccard: 0.1117 ± 0.0131"
echo "     Architecture: ResUNet"
echo ""
echo "  5. attention_resunet_lr5e-05_drop0.2_bs4"
echo "     Mean Jaccard: 0.1091 ± 0.0064"
echo "     Architecture: Attention ResUNet"
echo ""
echo "Using fold 1 models for each configuration"
echo "=============================="
echo ""

# =======================================================================
# OUTPUT CONFIGURATION
# =======================================================================

echo "=== OUTPUT CONFIGURATION ==="
echo "Test images: ./test_images/"
echo "Output: ./density_analysis_512_grayscale_YYYYMMDD_HHMMSS/"
echo ""
echo "Will generate:"
echo "  - 5 individual boxplots (boxplots/*.png)"
echo "    • unet_best_density_vs_dilution.png"
echo "    • unet_lr5e-05_d0.2_density_vs_dilution.png"
echo "    • unet_lr5e-05_d0.3_density_vs_dilution.png"
echo "    • resunet_density_vs_dilution.png"
echo "    • attention_resunet_density_vs_dilution.png"
echo ""
echo "  - 4-panel tile comparisons (representative_tiles/*.png)"
echo "    • Format: Original | U-Net (best) | U-Net (2) | U-Net (3)"
echo "    • 5 tiles per dilution factor"
echo ""
echo "  - CSV: All density data (csv_data/density_analysis_all_models.csv)"
echo "=============================="
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
if [ ! -f "./density_analysis_512_grayscale.py" ]; then
    echo "   ✗ ERROR: density_analysis_512_grayscale.py not found!"
    exit 1
fi
echo "   ✓ density_analysis_512_grayscale.py found"

if [ ! -f "./loss_functions_fixed.py" ]; then
    echo "   ✗ ERROR: loss_functions_fixed.py not found!"
    exit 1
fi
echo "   ✓ loss_functions_fixed.py found"

echo ""
echo "2. Checking model directory..."
if [ ! -d "./hyperparameter_search_512_20251014_235755/" ]; then
    echo "   ✗ ERROR: Model directory not found!"
    echo "   Expected: ./hyperparameter_search_512_20251014_235755/"
    exit 1
fi

# Check if models exist
model_count=$(find ./hyperparameter_search_512_20251014_235755/ -name "*_fold1_*_model.keras" 2>/dev/null | wc -l)
echo "   ✓ Model directory found"
echo "   ✓ Found $model_count model files"

if [ $model_count -lt 5 ]; then
    echo "   ⚠ WARNING: Expected at least 5 models, found $model_count"
fi

echo ""
echo "3. Checking test images..."
if [ ! -d "./test_images/" ]; then
    echo "   ✗ ERROR: test_images directory not found!"
    exit 1
fi

test_img_count=$(find ./test_images/ -name "*.tif" -o -name "*.tiff" 2>/dev/null | wc -l)
echo "   ✓ Test images directory found"
echo "   ✓ Found $test_img_count test images"

if [ $test_img_count -eq 0 ]; then
    echo "   ✗ ERROR: No .tif/.tiff files found in test_images/"
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
import sys

print('Python version:', sys.version.split()[0])
print('TensorFlow version:', tf.__version__)
print('CUDA built support:', tf.test.is_built_with_cuda())

# List GPUs
gpus = tf.config.list_physical_devices('GPU')
print('Physical GPUs found:', len(gpus))
for i, gpu in enumerate(gpus):
    print(f'  GPU {i}: {gpu}')

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
    print('⚠ WARNING: No GPU detected!')

# Check dependencies
print()
print('Checking dependencies:')
deps = ['cv2', 'PIL', 'matplotlib', 'numpy', 'pandas', 'seaborn', 'tqdm']
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
# EXECUTE DENSITY ANALYSIS
# =======================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🚀 STARTING DENSITY ANALYSIS"
echo "=============================================="
echo "Phase 1: Load top 5 models"
echo "Phase 2: Extract 512×512 tiles from test images"
echo "Phase 3: Predict on all tiles with all models"
echo "Phase 4: Calculate densities for each dilution"
echo "Phase 5: Generate visualizations"
echo "  → 5 individual boxplots"
echo "  → 4-panel tile comparisons"
echo ""
echo "Progress logged below..."
echo "=============================================="
echo ""

singularity exec --nv "$image" python3 density_analysis_512_grayscale.py 2>&1 | tee "density_analysis_512_console_${TIMESTAMP}.log"

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
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "density_analysis_512_grayscale_*" -mtime -1 | sort | tail -1)

    if [ -d "$OUTPUT_DIR" ]; then
        echo "📁 RESULTS DIRECTORY: $OUTPUT_DIR"
        echo ""

        # Copy logs and scripts to output directory
        echo "📋 Archiving logs and scripts..."
        if [ -n "$PBS_JOBID" ]; then
            PBS_OUTPUT_FILE="${PBS_JOBNAME}.o${PBS_JOBID}"
            if [ -f "$PBS_OUTPUT_FILE" ]; then
                cp "$PBS_OUTPUT_FILE" "$OUTPUT_DIR/"
                echo "   ✓ Copied PBS output: $PBS_OUTPUT_FILE"
            fi
        fi

        if [ -f "density_analysis_512_console_${TIMESTAMP}.log" ]; then
            cp "density_analysis_512_console_${TIMESTAMP}.log" "$OUTPUT_DIR/"
            echo "   ✓ Copied console log"
        fi

        cp density_analysis_512_grayscale.py "$OUTPUT_DIR/"
        cp pbs_density_analysis_512.sh "$OUTPUT_DIR/"
        echo "   ✓ Copied source scripts"
        echo ""

        # Check results
        echo "📊 GENERATED FILES:"

        if [ -d "$OUTPUT_DIR/boxplots" ]; then
            plot_count=$(ls "$OUTPUT_DIR/boxplots"/*.png 2>/dev/null | wc -l)
            echo "   Boxplots: $plot_count individual files in boxplots/"
            ls "$OUTPUT_DIR/boxplots"/*.png 2>/dev/null | sed 's/^/     - /'
        fi

        if [ -d "$OUTPUT_DIR/representative_tiles" ]; then
            tile_count=$(ls "$OUTPUT_DIR/representative_tiles"/*.png 2>/dev/null | wc -l)
            echo "   Representative tiles: $tile_count 4-panel comparisons"
            echo "     Format: Original | U-Net (best) | U-Net (2) | U-Net (3)"
        fi

        if [ -d "$OUTPUT_DIR/csv_data" ]; then
            csv_count=$(ls "$OUTPUT_DIR/csv_data"/*.csv 2>/dev/null | wc -l)
            echo "   CSV files: $csv_count in csv_data/"
            ls "$OUTPUT_DIR/csv_data"/*.csv 2>/dev/null | sed 's/^/     - /'
        fi

        echo ""
        echo "🎯 NEXT STEPS:"
        echo "============="
        echo "1. View individual boxplots:"
        echo "   Display: $OUTPUT_DIR/boxplots/*.png"
        echo ""
        echo "2. View representative 4-panel tiles:"
        echo "   Display: $OUTPUT_DIR/representative_tiles/*_comparison.png"
        echo ""
        echo "3. Analyze CSV data:"
        echo "   cat $OUTPUT_DIR/csv_data/density_analysis_all_models.csv"
        echo ""
        echo "4. Compare with 256×256 results:"
        echo "   Compare with: density_prediction_256_20251014_054939/"

    else
        echo "⚠ WARNING: Expected output directory not found!"
    fi

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Density analysis failed!"
    echo ""

    if [ -f "density_analysis_512_console_${TIMESTAMP}.log" ]; then
        echo "Last 50 lines of console log:"
        echo "-----------------------------"
        tail -50 "density_analysis_512_console_${TIMESTAMP}.log"
    fi

    echo ""
    echo "🔧 COMMON ISSUES:"
    echo "================="
    echo "1. Model loading errors:"
    echo "   - Check that model files exist in hyperparameter_search_512_20251014_235755/"
    echo "   - Verify model naming: {arch}_fold1_lr{val}_drop{val}_bs{val}_model.keras"
    echo ""
    echo "2. Test image issues:"
    echo "   - Check that test_images/ contains .tif or .tiff files"
    echo "   - Verify filenames contain dilution factors (e.g., 10x, 20x, etc.)"
    echo "   - Test images must be larger than 512×512 for tile extraction"
    echo ""
    echo "3. GPU memory issues:"
    echo "   - Check nvidia-smi for available memory"
    echo "   - Reduce batch size in CONFIG if needed"
    echo ""
    echo "4. Tile extraction issues:"
    echo "   - Verify test images are larger than 512×512"
    echo "   - Check image format and channel count"
fi

echo ""
echo "📝 CONSOLE LOG: density_analysis_512_console_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "DENSITY ANALYSIS JOB COMPLETE"
echo "512×512 Grayscale Models"
echo "======================================="
