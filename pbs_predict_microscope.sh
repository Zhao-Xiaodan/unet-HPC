#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Microscope_Prediction
#PBS -l select=1:ncpus=12:ngpus=1:mem=64gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# MICROSCOPE IMAGE PREDICTION - HPC PBS SCRIPT
# =======================================================================
# Runs inference on test images using trained models
# Handles large images (3840×2160) via tiling strategy
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "MICROSCOPE IMAGE SEGMENTATION - PREDICTION"
echo "======================================================================="
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# =======================================================================
# USER CONFIGURATION
# =======================================================================

# INPUT/OUTPUT DIRECTORIES
INPUT_DIR="./test_image"                              # Directory with images to predict
OUTPUT_DIR="./predictions_$(date +%Y%m%d_%H%M%S)"    # Timestamped output directory
MODEL_DIR="./microscope_training_20251008_074915"    # Training directory with models

# PREDICTION PARAMETERS
TILE_SIZE=256        # Must match training (default: 256)
OVERLAP=32           # Overlap between tiles (32=default, 64=higher quality)
THRESHOLD=0.5        # Binary threshold (0.5=default, try 0.3-0.7 if issues)

echo "=== CONFIGURATION ==="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Model directory: $MODEL_DIR"
echo "Tile size: ${TILE_SIZE}x${TILE_SIZE}"
echo "Tile overlap: ${OVERLAP}px"
echo "Binary threshold: $THRESHOLD"
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
    echo "Available containers:"
    ls -la /app1/common/singularity-img/hopper/tensorflow/
    exit 1
fi

echo "TensorFlow Container: $image"
echo ""

# =======================================================================
# PRE-EXECUTION CHECKS
# =======================================================================

echo "=== PRE-EXECUTION CHECKS ==="

# Check 1: Input directory exists
echo "1. Checking input directory..."
if [ -d "$INPUT_DIR" ]; then
    echo "   ✓ Input directory found: $INPUT_DIR"

    # Count images
    img_count=$(find "$INPUT_DIR" -type f \( -name "*.tif" -o -name "*.tiff" -o -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.bmp" \) | wc -l)

    echo "   ✓ Images found: $img_count files"

    if [ $img_count -eq 0 ]; then
        echo "   ERROR: No image files found in $INPUT_DIR"
        echo "   Supported formats: .tif, .tiff, .png, .jpg, .jpeg, .bmp"
        exit 1
    fi

    # Show sample files
    echo "   Sample files:"
    find "$INPUT_DIR" -type f \( -name "*.tif" -o -name "*.png" -o -name "*.jpg" \) | head -5 | sed 's/^/     /'

else
    echo "   ERROR: Input directory not found: $INPUT_DIR"
    echo ""
    echo "   Please create the directory and add images:"
    echo "     mkdir -p $INPUT_DIR"
    echo "     # Copy your test images to $INPUT_DIR/"
    exit 1
fi

# Check 2: Model directory exists
echo ""
echo "2. Checking model directory..."
if [ -d "$MODEL_DIR" ]; then
    echo "   ✓ Model directory found: $MODEL_DIR"

    # Check for model files
    model_count=$(find "$MODEL_DIR" -name "*.hdf5" -o -name "*.h5" | wc -l)
    echo "   ✓ Model files found: $model_count"

    if [ $model_count -eq 0 ]; then
        echo "   ERROR: No model files (.hdf5 or .h5) found in $MODEL_DIR"
        exit 1
    fi

    # List available models
    echo "   Available models:"
    find "$MODEL_DIR" -name "*.hdf5" -o -name "*.h5" | sed 's/^/     /'

else
    echo "   ERROR: Model directory not found: $MODEL_DIR"
    echo ""
    echo "   Available training directories:"
    ls -d microscope_training_* 2>/dev/null || echo "     None found"
    exit 1
fi

# Check 3: Python script exists
echo ""
echo "3. Checking prediction script..."
if [ -f "./predict_microscope.py" ]; then
    echo "   ✓ Prediction script found"
else
    echo "   ERROR: predict_microscope.py not found in current directory"
    echo "   Expected: ./predict_microscope.py"
    ls -la ./*.py | grep predict
    exit 1
fi

# Check 4: Required dependencies
echo ""
echo "4. Checking required files..."
required_files=("models.py" "focal_loss.py")
missing_files=0

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file (may be created automatically)"
        # Don't fail yet - focal_loss.py might be installed via pip
    fi
done

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

# Test basic GPU operation
if gpus:
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([1.0, 2.0])
            b = tf.constant([2.0, 3.0])
            c = tf.add(a, b)
        print('✓ Basic GPU operation successful')
    except Exception as e:
        print('GPU operation failed:', e)
else:
    print('WARNING: No GPUs detected. Prediction will run on CPU (slower).')

# Check key dependencies
print()
print('Checking dependencies:')
deps_to_check = ['cv2', 'PIL', 'matplotlib', 'numpy', 'argparse', 'pathlib']
all_ok = True
for dep in deps_to_check:
    try:
        __import__(dep)
        print(f'  ✓ {dep}')
    except ImportError:
        print(f'  ✗ {dep} - Missing!')
        all_ok = False

# Check focal_loss
try:
    from focal_loss import BinaryFocalLoss
    print('  ✓ focal_loss')
except ImportError:
    print('  ℹ focal_loss - Will attempt installation or use local file')

if all_ok:
    print()
    print('✓ All critical dependencies available')
else:
    print()
    print('⚠ Some dependencies missing - may cause errors')
"
echo "==============================="
echo ""

# =======================================================================
# INSTALL/CHECK FOCAL LOSS
# =======================================================================

echo "=== CHECKING FOCAL LOSS ==="

# Check if focal_loss.py exists locally
if [ -f "./focal_loss.py" ]; then
    echo "✓ Local focal_loss.py found"
else
    echo "Attempting to create focal_loss.py..."

    singularity exec --nv "$image" python3 -c "
import sys
import subprocess

# Try to import focal_loss
try:
    from focal_loss import BinaryFocalLoss
    print('✓ focal_loss package already available')
    sys.exit(0)
except ImportError:
    pass

# Try to install
print('Installing focal_loss...')
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'focal-loss', '--user', '--quiet'])
    print('✓ focal_loss installed successfully')
    sys.exit(0)
except:
    pass

# Create local implementation
print('Creating local focal_loss.py...')
with open('focal_loss.py', 'w') as f:
    f.write('''import tensorflow as tf
from tensorflow.keras import backend as K

class BinaryFocalLoss:
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal_loss = -alpha_t * K.pow(1 - pt, self.gamma) * K.log(pt)
        return K.mean(focal_loss)
''')
print('✓ Created local focal_loss.py')
"

    if [ -f "./focal_loss.py" ]; then
        echo "✓ focal_loss.py created successfully"
    fi
fi

echo "==========================="
echo ""

# =======================================================================
# CREATE OUTPUT DIRECTORY
# =======================================================================

mkdir -p "$OUTPUT_DIR"
echo "Created output directory: $OUTPUT_DIR"
echo ""

# =======================================================================
# RUN PREDICTION
# =======================================================================

echo "🚀 STARTING PREDICTION"
echo "=============================================="
echo "Processing images from: $INPUT_DIR"
echo "Saving results to: $OUTPUT_DIR"
echo "Using model from: $MODEL_DIR"
echo ""
echo "Tile configuration:"
echo "  - Size: ${TILE_SIZE}x${TILE_SIZE}"
echo "  - Overlap: ${OVERLAP}px"
echo "  - Threshold: $THRESHOLD"
echo ""
echo "This may take several minutes for large images..."
echo "=============================================="
echo ""

# Run prediction with all parameters
singularity exec --nv "$image" python3 predict_microscope.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_dir "$MODEL_DIR" \
    --tile_size "$TILE_SIZE" \
    --overlap "$OVERLAP" \
    --threshold "$THRESHOLD" \
    2>&1 | tee "${OUTPUT_DIR}/prediction_log.txt"

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
    echo "✓ Prediction completed successfully!"
    echo ""

    # Count output files
    mask_count=$(find "$OUTPUT_DIR/masks" -name "*.png" 2>/dev/null | wc -l)
    overlay_count=$(find "$OUTPUT_DIR/overlays" -name "*.png" 2>/dev/null | wc -l)
    comparison_count=$(find "$OUTPUT_DIR/comparisons" -name "*.png" 2>/dev/null | wc -l)

    echo "📊 OUTPUT SUMMARY:"
    echo "=================="
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  - Binary masks: $mask_count files (in $OUTPUT_DIR/masks/)"
    echo "  - Overlays: $overlay_count files (in $OUTPUT_DIR/overlays/)"
    echo "  - Comparisons: $comparison_count files (in $OUTPUT_DIR/comparisons/)"
    echo ""

    if [ -f "$OUTPUT_DIR/prediction_summary.txt" ]; then
        echo "📝 PREDICTION SUMMARY:"
        echo "====================="
        cat "$OUTPUT_DIR/prediction_summary.txt"
        echo ""
    fi

    echo "📁 RESULT LOCATIONS:"
    echo "===================="
    echo "  Binary masks:   $OUTPUT_DIR/masks/"
    echo "  Overlays:       $OUTPUT_DIR/overlays/"
    echo "  Comparisons:    $OUTPUT_DIR/comparisons/"
    echo "  Summary:        $OUTPUT_DIR/prediction_summary.txt"
    echo "  Full log:       $OUTPUT_DIR/prediction_log.txt"
    echo ""

    echo "🔍 TO DOWNLOAD RESULTS:"
    echo "======================="
    echo "From your local machine, run:"
    echo "  scp -r phyzxi@hpc:~/scratch/unet-HPC/$OUTPUT_DIR ./local_predictions/"
    echo ""

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Prediction failed!"
    echo ""
    echo "🔍 ERROR ANALYSIS:"
    echo "=================="

    if [ -f "$OUTPUT_DIR/prediction_log.txt" ]; then
        echo "Last 30 lines of log:"
        tail -30 "$OUTPUT_DIR/prediction_log.txt"
    fi
    echo ""

    echo "🔧 COMMON ISSUES:"
    echo "================="
    echo "1. Input directory empty: Check $INPUT_DIR has image files"
    echo "2. Model not found: Verify $MODEL_DIR contains .hdf5 files"
    echo "3. Out of memory: Try reducing tile_size or overlap"
    echo "4. Missing dependencies: Check focal_loss.py and models.py exist"
    echo "5. GPU issues: Job will still work on CPU (slower)"
fi

echo ""
echo "======================================================================="
echo "JOB COMPLETE - MICROSCOPE PREDICTION"
echo "======================================================================="
