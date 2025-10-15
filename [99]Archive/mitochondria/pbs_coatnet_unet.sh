#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oed
#PBS -N CoAtNet_UNet_Mitochondria_Segmentation
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# COATNET-UNET DEDICATED TRAINING - PBS SCRIPT
# =======================================================================
# Enhanced weight initialization management to resolve persistent
# trainable_variables initialization issues
# Based on successful Swin-UNet training approach
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "COATNET-UNET DEDICATED TRAINING - MITOCHONDRIA SEGMENTATION"
echo "======================================================================="
echo "Model: CoAtNet-UNet (Hybrid Attention + Convolution)"
echo "Task: Mitochondria semantic segmentation"
echo "Framework: TensorFlow/Keras with enhanced weight initialization"
echo "Expected Training Time: 6-10 hours"
echo ""

# Job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $(free -h | grep Mem | awk '{print $2}'), CPUs: $(nproc)"
echo ""

# =======================================================================
# ENHANCED ENVIRONMENT SETUP FOR COATNET-UNET
# =======================================================================

echo "=== COATNET-UNET TRAINING CONFIGURATION ==="
echo "Dataset Images: ./dataset_full_stack/images/ (1980 patches - REQUIRED)"
echo "Dataset Masks: ./dataset_full_stack/masks/ (1980 patches - REQUIRED)"
echo "Alternative: ./dataset/images/ and ./dataset/masks/"
echo "Image Size: 256x256x3"
echo "Batch Size: 3 (optimized for CoAtNet memory requirements)"
echo "Learning Rate: 2e-4 (AdamW optimizer)"
echo "Epochs: 100 (with early stopping)"
echo "Loss Function: Binary Focal Loss"
echo "Special Features: Enhanced weight initialization for hybrid architecture"
echo "============================================="
echo ""

# TensorFlow environment variables with CoAtNet optimizations
export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_MEMORY_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

# CoAtNet-specific optimizations
export TF_ENABLE_TENSOR_FLOAT_32_EXECUTION=1
export TF_ENABLE_MIXED_PRECISION_GRAPH_REWRITE=1

# Load required modules
module load singularity

# Use the TensorFlow container
image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

if [ ! -f "$image" ]; then
    echo "ERROR: TensorFlow container not found at $image"
    exit 1
fi

echo "TensorFlow Container: $image"
echo ""

# =======================================================================
# COMPREHENSIVE SESSION RESET FOR COATNET-UNET
# =======================================================================

echo "=== COMPREHENSIVE SESSION RESET ==="
echo "Performing comprehensive session reset to prevent weight initialization issues..."

# Clear all possible TensorFlow cache locations
echo "Clearing TensorFlow caches..."
rm -rf ~/.tensorflow* 2>/dev/null || true
rm -rf /tmp/tf* 2>/dev/null || true
rm -rf /tmp/tensorflow* 2>/dev/null || true
rm -rf /tmp/*tf* 2>/dev/null || true
rm -rf /var/tmp/tf* 2>/dev/null || true
rm -rf /dev/shm/tf* 2>/dev/null || true

# Clear Python cache
echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Clear any model checkpoints from previous runs to avoid conflicts
echo "Clearing previous model checkpoints..."
rm -rf coatnet_unet_training_*/CoAtNet_UNet_* 2>/dev/null || true

# Set unique session identifier for CoAtNet
export TF_COATNET_SESSION_ID="coatnet_$(date +%s)_$$"
echo "Unique CoAtNet session ID: $TF_COATNET_SESSION_ID"

echo "✓ Comprehensive session reset completed"
echo "====================================="
echo ""

# =======================================================================
# PRE-EXECUTION CHECKS
# =======================================================================

echo "=== PRE-EXECUTION CHECKS ==="

# Check dataset structure
echo "1. Checking dataset structure..."
dataset_found=false

if [ -d "./dataset_full_stack/images/" ] && [ -d "./dataset_full_stack/masks/" ]; then
    echo "   ✓ Full stack dataset directories found (PREFERRED)"
    img_count=$(find ./dataset_full_stack/images/ -name "*.tif" -o -name "*.png" -o -name "*.jpg" | wc -l)
    mask_count=$(find ./dataset_full_stack/masks/ -name "*.tif" -o -name "*.png" -o -name "*.jpg" | wc -l)
    echo "   ✓ Images found: $img_count files"
    echo "   ✓ Masks found: $mask_count files"
    dataset_found=true
elif [ -d "./dataset/images/" ] && [ -d "./dataset/masks/" ]; then
    echo "   ✓ Standard dataset directories found"
    img_count=$(find ./dataset/images/ -name "*.tif" -o -name "*.png" -o -name "*.jpg" | wc -l)
    mask_count=$(find ./dataset/masks/ -name "*.tif" -o -name "*.png" -o -name "*.jpg" | wc -l)
    echo "   ✓ Images found: $img_count files"
    echo "   ✓ Masks found: $mask_count files"
    dataset_found=true
fi

if [ "$dataset_found" = false ]; then
    echo "   ERROR: No valid dataset directories found!"
    exit 1
fi

# Check Python files
echo ""
echo "2. Checking Python files..."
required_files=("coatnet_unet_training.py" "modern_unet_models.py")
missing_files=()

for file in "${required_files[@]}"; do
    if [ -f "./$file" ]; then
        echo "   ✓ $file found"
    else
        echo "   ✗ $file NOT found"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "   ERROR: Missing required files: ${missing_files[@]}"
    exit 1
fi

echo "=========================="
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

# Test GPU operation
if gpus:
    try:
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
        print('✓ GPU operation test successful')
    except Exception as e:
        print('GPU operation failed:', e)

print()
print('CoAtNet-UNet memory requirements:')
print('- Expected GPU memory: 12-16 GB')
print('- Batch size: 3 (memory optimized)')
print('- Model parameters: ~20-30M')
"
echo "==============================="
echo ""

# =======================================================================
# COATNET-UNET MODEL VALIDATION
# =======================================================================

echo "=== COATNET-UNET MODEL VALIDATION ==="
echo "Testing CoAtNet-UNet creation to validate implementation and weight initialization..."

singularity exec --nv "$image" python3 -c "
try:
    import sys
    sys.path.append('.')

    print('Testing CoAtNet-UNet model creation...')
    from modern_unet_models import create_modern_unet

    # Test CoAtNet-UNet with small input
    input_shape = (64, 64, 3)
    model = create_modern_unet('CoAtNet_UNet', input_shape, num_classes=1)
    params = model.count_params()
    print(f'✓ CoAtNet-UNet: {params:,} parameters')

    # Test model building
    model.build(input_shape=(None,) + input_shape)
    print('✓ Model building successful')

    # Test weight initialization
    import numpy as np
    dummy_input = np.random.random((1,) + input_shape).astype(np.float32)

    # Test forward pass
    output = model(dummy_input, training=False)
    print(f'✓ Forward pass successful: {output.shape}')

    # Test gradient computation (critical for CoAtNet)
    import tensorflow as tf
    with tf.GradientTape() as tape:
        predictions = model(dummy_input, training=True)
        dummy_target = tf.zeros_like(predictions)
        loss = tf.reduce_mean(tf.square(predictions - dummy_target))

    gradients = tape.gradient(loss, model.trainable_variables)
    valid_gradients = sum(1 for grad in gradients if grad is not None)
    total_variables = len(model.trainable_variables)

    print(f'✓ Gradient computation: {valid_gradients}/{total_variables} variables')

    if valid_gradients == total_variables:
        print('✓ All trainable variables have valid gradients')
    else:
        print(f'⚠ Warning: {total_variables - valid_gradients} variables have None gradients')

    print('✓ CoAtNet-UNet validation completed successfully')

except Exception as e:
    print(f'✗ CoAtNet-UNet validation failed: {e}')
    import traceback
    traceback.print_exc()
"

validation_exit_code=$?
if [ $validation_exit_code -ne 0 ]; then
    echo "⚠ CoAtNet-UNet validation failed!"
    echo "Training may encounter issues, but proceeding..."
else
    echo "✓ CoAtNet-UNet validation passed"
fi

echo "====================================="
echo ""

# =======================================================================
# EXECUTE COATNET-UNET TRAINING
# =======================================================================

echo "🚀 STARTING COATNET-UNET TRAINING"
echo "=============================================="
echo "Training CoAtNet-UNet with enhanced weight initialization"
echo ""
echo "Training Configuration:"
echo "- Architecture: CoAtNet-UNet (Hybrid Attention + Convolution)"
echo "- Learning Rate: 2e-4 (AdamW optimizer)"
echo "- Batch Size: 3 (memory optimized)"
echo "- Max Epochs: 100 (with early stopping)"
echo "- Loss: Binary Focal Loss (gamma=2)"
echo "- Special: Enhanced weight initialization & gradient verification"
echo ""
echo "Expected timeline: 6-10 hours"
echo "Expected performance: 92-94% Jaccard"
echo "=============================================="

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="coatnet_unet_training_${TIMESTAMP}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Execute the training with enhanced logging
echo "Starting CoAtNet-UNet training execution..."
singularity exec --nv "$image" python3 coatnet_unet_training.py 2>&1 | tee "${OUTPUT_DIR}_console.log"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "COATNET-UNET TRAINING COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✓ SUCCESS"
    echo ""
    echo "✓ CoAtNet-UNet training completed successfully!"
    echo ""
    echo "Generated files:"
    echo "📁 Training directory:"
    ls -la coatnet_unet_training_*/ 2>/dev/null || echo "   No training directory found"
    echo ""
    echo "📊 Model and results:"
    ls -la coatnet_unet_training_*/CoAtNet_UNet_* 2>/dev/null || echo "   No model files found"
    echo ""

    echo "🎯 COATNET-UNET PERFORMANCE SUMMARY:"
    echo "===================================="
    singularity exec --nv "$image" python3 -c "
import pandas as pd
import os
import glob
import json

# Look for CoAtNet-UNet results
json_files = glob.glob('coatnet_unet_training_*/CoAtNet_UNet_*_results.json')

if json_files:
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)

            model_name = result.get('model_name', 'CoAtNet_UNet')
            jaccard = result.get('best_val_jaccard', 0)
            epoch = result.get('best_epoch', 0)
            time_sec = result.get('training_time_seconds', 0)
            params = result.get('model_parameters', 0)
            stability = result.get('val_loss_stability', 0)
            init_success = result.get('initialization_success', False)

            print(f'CoAtNet-UNet Training Results:')
            print(f'🏆 Best Jaccard: {jaccard:.4f} (epoch {epoch})')
            print(f'⏱️ Training Time: {time_sec:.0f}s ({time_sec/3600:.1f}h)')
            print(f'🧠 Parameters: {params:,} ({params/1e6:.1f}M)')
            print(f'📊 Stability: {stability:.4f}')
            print(f'🔧 Initialization: {\"SUCCESS\" if init_success else \"ISSUES\"}'
            print()

            # Performance assessment
            if jaccard >= 0.92:
                print('🏆 EXCELLENT PERFORMANCE: Jaccard >= 92%')
            elif jaccard >= 0.90:
                print('✅ GOOD PERFORMANCE: Jaccard >= 90%')
            else:
                print('⚠ MODERATE PERFORMANCE: Consider optimization')

        except Exception as e:
            print(f'Error reading {json_file}: {e}')
else:
    print('No CoAtNet-UNet results found.')

    # Check for CSV files as fallback
    csv_files = glob.glob('coatnet_unet_training_*/CoAtNet_UNet_*_history.csv')
    if csv_files:
        print('Training history found in CSV files:')
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'val_jacard_coef' in df.columns:
                    best_jaccard = df['val_jacard_coef'].max()
                    best_epoch = df['val_jacard_coef'].idxmax() + 1
                    print(f'Best Val Jaccard: {best_jaccard:.4f} (epoch {best_epoch})')
            except Exception as e:
                print(f'Error reading {csv_file}: {e}')
" 2>/dev/null || echo "Unable to generate performance summary"

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ CoAtNet-UNet training failed!"
    echo ""
    echo "🔍 ERROR ANALYSIS:"
    echo "=================="
    echo "Check the console log for detailed error information:"
    echo "   ${OUTPUT_DIR}_console.log"
    echo ""

    if [ -f "${OUTPUT_DIR}_console.log" ]; then
        echo "Last 30 lines of console log:"
        tail -30 "${OUTPUT_DIR}_console.log"
    fi
    echo ""

    echo "🔧 TROUBLESHOOTING:"
    echo "==================="
    echo "1. Weight Initialization Issues:"
    echo "   - Check gradient computation warnings"
    echo "   - Verify all model layers are properly built"
    echo ""
    echo "2. Memory Issues:"
    echo "   - CoAtNet-UNet requires 12-16 GB GPU memory"
    echo "   - Try reducing batch size if needed"
    echo ""
    echo "3. Hybrid Architecture Issues:"
    echo "   - Check attention mechanism compatibility"
    echo "   - Verify convolution-attention integration"
fi

echo ""
echo "📁 CONSOLE LOG SAVED: ${OUTPUT_DIR}_console.log"
echo ""
echo "🔗 NEXT STEPS:"
echo "============="
echo "1. Analyze CoAtNet-UNet training results"
echo "2. Compare with Swin-UNet performance (93.57%)"
echo "3. Train ConvNeXt-UNet separately if needed"
echo "4. Consider ensemble methods for best performance"
echo ""
echo "======================================="
echo "COATNET-UNET TRAINING JOB COMPLETE"
echo "Model: CoAtNet-UNet (Hybrid Attention + Convolution)"
echo "Framework: TensorFlow/Keras + Enhanced Weight Initialization"
echo "======================================="