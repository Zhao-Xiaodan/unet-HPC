#!/bin/bash
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Density_Analysis
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# DENSITY ANALYSIS - ARCHITECTURE COMPARISON
# =======================================================================
# Trains models for U-Net, ResUNet, and Attention ResUNet using
# configurations from validation_arch_comparison_20251013_093844,
# then performs density analysis on test images with dilution series.
#
# Generates:
# - 4 PNG plots (one per architecture/method)
# - 1 comprehensive CSV with all density data
# - Y-axis: Foreground Percentage (log scale)
# - X-axis: 1/Dilution Factor
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "DENSITY ANALYSIS - ARCHITECTURE COMPARISON"
echo "======================================================================="
echo "Training models and analyzing density across dilution series"
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

echo "=== ANALYSIS CONFIGURATION ==="
echo "Architectures: U-Net, ResUNet, Attention ResUNet, CLAHE+OTSU"
echo "Training Dataset: ./dataset_full_stack/"
echo "Test Images: ./test_images/ (dilution series)"
echo ""
echo "Model Configuration (from validation_arch_comparison):"
echo "  Filters: 64"
echo "  Dropout: 0.2"
echo "  Batch Size: 16"
echo "  Learning Rate: 5e-5"
echo "  Loss: Combined Dice + Focal"
echo "  Max Epochs: 50"
echo "  Early Stopping Patience: 10"
echo ""
echo "Analysis Configuration:"
echo "  Tile Size: 256×256"
echo "  Metric: Foreground Percentage (log scale)"
echo "  Grouping: By dilution factor (10x, 20x, 80x, 160x, etc.)"
echo ""
echo "Output:"
echo "  - unet_density_vs_dilution.png"
echo "  - resunet_density_vs_dilution.png"
echo "  - attention_resunet_density_vs_dilution.png"
echo "  - clahe_otsu_density_vs_dilution.png"
echo "  - density_analysis_comprehensive.csv"
echo ""
echo "Expected Runtime: 4-6 hours"
echo "  - Model training: ~3-4 hours (3 models × ~1 hour each)"
echo "  - Prediction & analysis: ~1-2 hours"
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
if [ ! -f "./density_analysis_arch_comparison.py" ]; then
    echo "   ✗ ERROR: density_analysis_arch_comparison.py not found!"
    echo "   Current directory: $(pwd)"
    ls -la ./*.py | head -10
    exit 1
fi
echo "   ✓ density_analysis_arch_comparison.py found"

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

# Check training dataset structure
echo ""
echo "2. Checking training dataset..."
if [ ! -d "./dataset_full_stack/images/" ] || [ ! -d "./dataset_full_stack/masks/" ]; then
    echo "   ✗ ERROR: Training dataset directories not found!"
    echo "   Expected: ./dataset_full_stack/images/ and ./dataset_full_stack/masks/"
    exit 1
fi

train_img_count=$(find ./dataset_full_stack/images/ -name "*.tif" 2>/dev/null | wc -l)
train_mask_count=$(find ./dataset_full_stack/masks/ -name "*.tif" 2>/dev/null | wc -l)

echo "   ✓ Training dataset directories found"
echo "   ✓ Training images: $train_img_count .tif files"
echo "   ✓ Training masks: $train_mask_count .tif files"

if [ $train_img_count -eq 0 ] || [ $train_mask_count -eq 0 ]; then
    echo "   ✗ ERROR: No .tif files found in training dataset"
    exit 1
fi

# Check test images directory
echo ""
echo "3. Checking test images..."
if [ ! -d "./test_images/" ]; then
    echo "   ✗ ERROR: Test images directory not found!"
    echo "   Expected: ./test_images/ (with dilution series)"
    exit 1
fi

test_img_count=$(find ./test_images/ -name "*.tif" 2>/dev/null | wc -l)

echo "   ✓ Test images directory found"
echo "   ✓ Test images: $test_img_count .tif files"

if [ $test_img_count -eq 0 ]; then
    echo "   ✗ ERROR: No .tif files found in test_images/"
    exit 1
fi

# Sample test image names (to verify dilution factors)
echo "   Sample test images:"
find ./test_images/ -name "*.tif" 2>/dev/null | head -5 | while read img; do
    echo "     - $(basename $img)"
done

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
    print('⚠ WARNING: No GPU detected! Training will be VERY slow.')

# Check key dependencies
print()
print('Checking dependencies:')
deps_to_check = ['cv2', 'PIL', 'matplotlib', 'numpy', 'sklearn', 'pandas', 'seaborn', 'tqdm']
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
OUTPUT_DIR="density_analysis_arch_comparison_${TIMESTAMP}"

echo "=== OUTPUT CONFIGURATION ==="
echo "Results directory: $OUTPUT_DIR"
echo "This will contain:"
echo "  - trained_models/ (best model for each architecture)"
echo "  - plots/ (4 PNG files)"
echo "  - csv_data/ (comprehensive density CSV)"
echo "=============================="
echo ""

# =======================================================================
# EXECUTE DENSITY ANALYSIS
# =======================================================================

echo "🚀 STARTING DENSITY ANALYSIS"
echo "=============================================="
echo "Phase 1: Train models (U-Net, ResUNet, Attention ResUNet)"
echo "Phase 2: Predict on test images with dilution series"
echo "Phase 3: Calculate foreground percentage per tile"
echo "Phase 4: Generate plots and comprehensive CSV"
echo ""
echo "Progress will be logged below..."
echo "=============================================="
echo ""

# Execute with logging
singularity exec --nv "$image" python3 density_analysis_arch_comparison.py 2>&1 | tee "density_analysis_console_${TIMESTAMP}.log"

# Capture exit code
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

    # Check if output directory was created
    if [ -d "$OUTPUT_DIR" ]; then
        echo "📁 RESULTS DIRECTORY: $OUTPUT_DIR"
        echo ""

        # Count generated files
        model_count=$(find "$OUTPUT_DIR/trained_models" -name "*.keras" 2>/dev/null | wc -l)
        plot_count=$(find "$OUTPUT_DIR/plots" -name "*.png" 2>/dev/null | wc -l)
        csv_count=$(find "$OUTPUT_DIR/csv_data" -name "*.csv" 2>/dev/null | wc -l)

        echo "📊 GENERATED FILES:"
        echo "  Models trained: $model_count"
        echo "  Plots created: $plot_count"
        echo "  CSV files: $csv_count"
        echo ""

        # List plot files
        if [ $plot_count -gt 0 ]; then
            echo "🖼️  PLOTS GENERATED:"
            find "$OUTPUT_DIR/plots" -name "*.png" -exec basename {} \; | sort
            echo ""
        fi

        # Check CSV file
        if [ -f "$OUTPUT_DIR/csv_data/density_analysis_comprehensive.csv" ]; then
            echo "📄 COMPREHENSIVE CSV:"
            echo "   $OUTPUT_DIR/csv_data/density_analysis_comprehensive.csv"

            # Show CSV summary
            singularity exec --nv "$image" python3 -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('$OUTPUT_DIR/csv_data/density_analysis_comprehensive.csv')

    print()
    print('CSV SUMMARY:')
    print('-' * 60)
    print(f'Total measurements: {len(df)}')
    print(f'Methods: {df[\"method\"].unique().tolist()}')
    print(f'Dilution factors: {sorted(df[\"dilution_factor\"].unique())}')
    print(f'Images analyzed: {df[\"image\"].nunique()}')
    print()

    # Show mean foreground percentage by method
    print('Mean Foreground Percentage by Method:')
    print('-' * 60)
    for method in df['method'].unique():
        mean_fg = df[df['method'] == method]['foreground_pct'].mean()
        print(f'  {method.ljust(20)}: {mean_fg:.4f}%')
    print()

except Exception as e:
    print(f'Unable to parse CSV: {e}')
" 2>/dev/null || echo "   Results saved. Check CSV for details."

        fi

    else
        echo "⚠ WARNING: Expected output directory not found!"
        echo "Searching for density_analysis_arch_comparison directories..."
        find . -maxdepth 1 -type d -name "density_analysis_arch_comparison_*" -mtime -1
    fi

    echo ""
    echo "🎯 NEXT STEPS:"
    echo "============="
    echo "1. Review generated plots:"
    echo "   $OUTPUT_DIR/plots/"
    echo ""
    echo "2. Analyze comprehensive CSV data:"
    echo "   $OUTPUT_DIR/csv_data/density_analysis_comprehensive.csv"
    echo ""
    echo "3. Compare architectures:"
    echo "   - Check which architecture matches CLAHE+OTSU baseline"
    echo "   - Examine density trends across dilution factors"
    echo "   - Identify outliers or anomalies"

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Density analysis failed!"
    echo ""
    echo "🔍 ERROR ANALYSIS:"
    echo "=================="
    echo "Check the console log for detailed error information:"
    echo "   density_analysis_console_${TIMESTAMP}.log"
    echo ""

    if [ -f "density_analysis_console_${TIMESTAMP}.log" ]; then
        echo "Last 30 lines of console log:"
        echo "-----------------------------"
        tail -30 "density_analysis_console_${TIMESTAMP}.log"
    fi
    echo ""

    echo "🔧 COMMON ISSUES AND SOLUTIONS:"
    echo "==============================="
    echo "1. Training dataset missing: Ensure ./dataset_full_stack/ exists with images and masks"
    echo "2. Test images missing: Ensure ./test_images/ exists with dilution series"
    echo "3. GPU memory issues: Models may be too large (try reducing batch_size in script)"
    echo "4. Import errors: Check model_architectures.py and loss_functions_fixed.py"
    echo "5. Timeout: Job may need more than 8 hours (increase walltime)"
    echo ""
fi

echo ""
echo "📝 CONSOLE LOG: density_analysis_console_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "DENSITY ANALYSIS JOB COMPLETE"
echo "Architectures: U-Net, ResUNet, Attention ResUNet, CLAHE+OTSU"
echo "Framework: TensorFlow/Keras"
echo "======================================="
