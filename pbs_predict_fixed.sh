#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Microbead_Prediction_Fixed
#PBS -l select=1:ncpus=12:ngpus=1:mem=64gb
#PBS -M phyzxi@nus.edu.sg

# =======================================================================
# FIXED MICROBEAD PREDICTION ON HPC
# =======================================================================
# Corrections:
# 1. Tile size = 512×512 (matches training at original resolution!)
# 2. Gaussian-weighted blending (no black grid lines)
# 3. Smooth transitions between tiles
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "FIXED MICROBEAD PREDICTION"
echo "======================================================================="
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo ""

# =======================================================================
# CONFIGURATION
# =======================================================================

# Input/output directories
INPUT_DIR=${INPUT_DIR:-"./test_images"}
OUTPUT_DIR=${OUTPUT_DIR:-"predictions_fixed_$(date +%Y%m%d_%H%M%S)"}

# Model path (update this to your best model!)
# Default: best UNet from microbead_training_20251009_073134
MODEL_PATH=${MODEL_PATH:-"./microbead_training_20251009_073134/best_unet_model.hdf5"}

# Tiling parameters
TILE_SIZE=512    # MUST match training size (using original resolution)!
OVERLAP=64       # Overlap for smooth blending (larger for 512×512)
THRESHOLD=0.5    # Prediction threshold

echo "=== CONFIGURATION ==="
echo "Model: $MODEL_PATH"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Tile size: ${TILE_SIZE}×${TILE_SIZE} (original 512×512 resolution ✓)"
echo "Overlap: $OVERLAP pixels (smooth blending ✓)"
echo "Threshold: $THRESHOLD"
echo "====================="
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
    exit 1
fi

echo "TensorFlow Container: $image"
echo ""

# =======================================================================
# PRE-EXECUTION CHECKS
# =======================================================================

echo "=== PRE-EXECUTION CHECKS ==="

# Check 1: Model file
echo "1. Checking model..."
if [ -f "$MODEL_PATH" ]; then
    echo "   ✓ Model found: $MODEL_PATH"
    echo "   Size: $(du -h "$MODEL_PATH" | cut -f1)"
else
    echo "   ERROR: Model not found at $MODEL_PATH"
    echo ""
    echo "   Available models:"
    find . -name "*.hdf5" -type f 2>/dev/null | head -10
    exit 1
fi

# Check 2: Input directory
echo ""
echo "2. Checking input directory..."
if [ -d "$INPUT_DIR" ]; then
    img_count=$(find "$INPUT_DIR" -type f \( -name "*.tif" -o -name "*.png" -o -name "*.jpg" \) | wc -l)
    echo "   ✓ Input directory found: $INPUT_DIR"
    echo "   ✓ Images found: $img_count"

    if [ $img_count -eq 0 ]; then
        echo "   WARNING: No images found in $INPUT_DIR"
        echo "   Supported formats: .tif, .png, .jpg"
    fi
else
    echo "   ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Check 3: Prediction script
echo ""
echo "3. Checking prediction script..."
if [ -f "./predict_microscope_fixed.py" ]; then
    echo "   ✓ Fixed prediction script found"
else
    echo "   ERROR: predict_microscope_fixed.py not found"
    echo "   Please ensure the file is uploaded to HPC"
    exit 1
fi

echo "============================="
echo ""

# =======================================================================
# GPU STATUS CHECK
# =======================================================================

echo "=== GPU STATUS ==="
singularity exec --nv "$image" python3 -c "
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {len(gpus)}')
for i, gpu in enumerate(gpus):
    print(f'  GPU {i}: {gpu.name}')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('✓ GPU memory growth enabled')
"
echo "=================="
echo ""

# =======================================================================
# RUN PREDICTION
# =======================================================================

echo "🔮 STARTING PREDICTION"
echo "=============================================="
echo ""

# Execute prediction
singularity exec --nv "$image" python3 predict_microscope_fixed.py \
    --model "$MODEL_PATH" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --tile-size $TILE_SIZE \
    --overlap $OVERLAP \
    --threshold $THRESHOLD

# Capture exit code
EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "PREDICTION COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✓ SUCCESS"
    echo ""

    if [ -d "$OUTPUT_DIR" ]; then
        mask_count=$(find "$OUTPUT_DIR" -name "*_mask.png" | wc -l)
        overlay_count=$(find "$OUTPUT_DIR" -name "*_overlay.png" | wc -l)

        echo "📊 PREDICTION RESULTS:"
        echo "====================="
        echo "  Masks generated: $mask_count"
        echo "  Overlays generated: $overlay_count"
        echo ""
        echo "📁 OUTPUT DIRECTORY: $OUTPUT_DIR"
        echo ""

        echo "📥 TO DOWNLOAD RESULTS:"
        echo "======================="
        echo "From your local machine:"
        echo "  scp -r phyzxi@hpc:~/scratch/unet-HPC/$OUTPUT_DIR ./predictions/"
        echo ""

        echo "✓ KEY IMPROVEMENTS IN THIS VERSION:"
        echo "====================================="
        echo "  ✓ Tile size = 512×512 (matches training at original resolution!)"
        echo "  ✓ Gaussian-weighted blending (no black grid lines)"
        echo "  ✓ Smooth transitions between overlapping tiles"
        echo "  ✓ Proper normalization of overlapping regions"
        echo ""

        # Show sample output files
        echo "Sample output files:"
        ls -lh "$OUTPUT_DIR" | head -10
    fi
else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "Check the output above for error messages"
fi

echo ""
echo "======================================================================="
