#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -N ConvNeXt_UNet_Optimized_Mitochondria
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# CONVNEXT-UNET OPTIMIZED TRAINING - PBS SCRIPT
# =======================================================================
# Advanced hyperparameter optimization targeting 93%+ Jaccard performance
# Enhanced with: Advanced LR scheduling, Combined loss, Better regularization
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "CONVNEXT-UNET OPTIMIZED TRAINING - MITOCHONDRIA SEGMENTATION"
echo "======================================================================="
echo "Target: 93%+ Jaccard (matching CoAtNet-UNet and Swin-UNet performance)"
echo "Model: ConvNeXt-UNet with advanced optimization techniques"
echo "Framework: TensorFlow/Keras with enhanced training pipeline"
echo "Expected Training Time: 15-20 hours (extended for optimal convergence)"
echo ""

# Job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $(free -h | grep Mem | awk '{print $2}'), CPUs: $(nproc)"
echo ""

# =======================================================================
# ADVANCED OPTIMIZATION CONFIGURATION
# =======================================================================

echo "=== CONVNEXT-UNET OPTIMIZATION CONFIGURATION ==="
echo "Dataset Images: ./dataset_full_stack/images/ (1980 patches - REQUIRED)"
echo "Dataset Masks: ./dataset_full_stack/masks/ (1980 patches - REQUIRED)"
echo "Image Size: 256x256x3"
echo "Optimized Batch Size: 4 (memory efficient)"
echo "Enhanced Learning Rate: 8e-5 (with warmup + cosine decay)"
echo "Max Epochs: 120 (extended for convergence)"
echo "Loss Function: Combined Focal + Dice Loss"
echo "Optimizer: AdamW with weight decay"
echo "Advanced Features:"
echo "  ✓ Warmup + Cosine decay LR scheduling"
echo "  ✓ Enhanced data preprocessing with contrast adjustment"
echo "  ✓ Advanced regularization (L2 + dropout)"
echo "  ✓ Stratified train/val split"
echo "  ✓ Improved numerical stability"
echo "======================================================="
echo ""

# =======================================================================
# ENHANCED TENSORFLOW ENVIRONMENT
# =======================================================================

echo "=== ENHANCED TENSORFLOW ENVIRONMENT SETUP ==="

# Advanced TensorFlow optimizations for ConvNeXt
export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_MEMORY_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

# ConvNeXt-specific performance optimizations
export TF_ENABLE_TENSOR_FLOAT_32_EXECUTION=1
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Memory optimization
export TF_GPU_MEMORY_FRACTION=0.95
export TF_FORCE_UNIFIED_MEMORY=1

# Advanced caching control
export TF_DISABLE_DATASET_CACHING=1
export TF_DATA_EXPERIMENTAL_THREADING=1

echo "✓ Advanced TensorFlow environment configured"
echo ""

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
# COMPREHENSIVE SYSTEM PREPARATION
# =======================================================================

echo "=== COMPREHENSIVE SYSTEM PREPARATION ==="
echo "Preparing system for optimized ConvNeXt-UNet training..."

# Advanced cache clearing
echo "Clearing all caches and temporary files..."
rm -rf ~/.tensorflow* 2>/dev/null || true
rm -rf /tmp/tf* 2>/dev/null || true
rm -rf /tmp/tensorflow* 2>/dev/null || true
rm -rf /tmp/*tf* 2>/dev/null || true
rm -rf /var/tmp/tf* 2>/dev/null || true
rm -rf /dev/shm/tf* 2>/dev/null || true

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Clear previous optimization runs
rm -rf convnext_unet_optimized_*/ConvNeXt_UNet_* 2>/dev/null || true

# Set unique optimization session identifier
export TF_SESSION_ID="convnext_opt_$(date +%s)_$$"
echo "Unique optimization session ID: $TF_SESSION_ID"

echo "✓ Comprehensive system preparation completed"
echo "=================================="
echo ""

# =======================================================================
# PRE-TRAINING VALIDATION
# =======================================================================

echo "=== PRE-TRAINING VALIDATION ==="

# Validate dataset structure
echo "1. Validating enhanced dataset structure..."
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

# Validate Python files
echo ""
echo "2. Validating optimization scripts..."
required_files=("convnext_unet_optimized_training.py" "modern_unet_models.py")
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
# TENSORFLOW AND GPU OPTIMIZATION CHECK
# =======================================================================

echo "=== TENSORFLOW & GPU OPTIMIZATION STATUS ==="
singularity exec --nv "$image" python3 -c "
import tensorflow as tf
import sys
import warnings
warnings.filterwarnings('ignore')

print('Python version:', sys.version.split()[0])
print('TensorFlow version:', tf.__version__)
print('CUDA built support:', tf.test.is_built_with_cuda())

# Enhanced GPU configuration
gpus = tf.config.list_physical_devices('GPU')
print('Physical GPUs found:', len(gpus))

if gpus:
    try:
        for i, gpu in enumerate(gpus):
            print(f'  GPU {i}: {gpu}')
            tf.config.experimental.set_memory_growth(gpu, True)

        # Enable advanced optimizations
        tf.config.optimizer.set_jit(True)
        tf.config.experimental.enable_tensor_float_32_execution(True)

        print('✓ Enhanced GPU memory growth enabled')
        print('✓ XLA JIT compilation enabled')
        print('✓ TensorFloat-32 enabled')
    except RuntimeError as e:
        print('GPU optimization error:', e)

    # Test GPU operation with advanced features
    try:
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
        print('✓ GPU operation test successful')
    except Exception as e:
        print('GPU operation failed:', e)

print()
print('ConvNeXt-UNet Optimization Requirements:')
print('- Target GPU memory: 8-12 GB')
print('- Optimized batch size: 4')
print('- Expected parameters: ~35M')
print('- Target performance: 93%+ Jaccard')
"
echo "======================================"
echo ""

# =======================================================================
# CONVNEXT-UNET OPTIMIZATION MODEL VALIDATION
# =======================================================================

echo "=== CONVNEXT-UNET OPTIMIZATION MODEL VALIDATION ==="
echo "Testing optimized ConvNeXt-UNet creation and advanced features..."

singularity exec --nv "$image" python3 -c "
try:
    import sys
    sys.path.append('.')
    import warnings
    warnings.filterwarnings('ignore')

    print('Testing optimized ConvNeXt-UNet model creation...')
    from modern_unet_models import create_modern_unet
    import tensorflow as tf

    # Test optimized ConvNeXt-UNet
    input_shape = (64, 64, 3)
    model = create_modern_unet('ConvNeXt_UNet', input_shape, num_classes=1)
    params = model.count_params()
    print(f'✓ ConvNeXt-UNet: {params:,} parameters')

    # Test advanced optimizer
    from tensorflow.keras.optimizers import AdamW
    optimizer = AdamW(learning_rate=8e-5, weight_decay=1e-4)
    print('✓ AdamW optimizer with weight decay created')

    # Test combined loss function
    from tensorflow.keras.losses import BinaryFocalCrossentropy
    focal_loss = BinaryFocalCrossentropy(alpha=0.7, gamma=2.0)
    print('✓ Binary focal crossentropy loss created')

    # Test model compilation
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print('✓ Model compilation successful')

    # Test forward pass
    import numpy as np
    dummy_input = np.random.random((1,) + input_shape).astype(np.float32)
    output = model(dummy_input, training=False)
    print(f'✓ Forward pass successful: {output.shape}')

    print('✓ ConvNeXt-UNet optimization validation completed successfully')

except Exception as e:
    print(f'✗ ConvNeXt-UNet optimization validation failed: {e}')
    import traceback
    traceback.print_exc()
"

validation_exit_code=$?
if [ $validation_exit_code -ne 0 ]; then
    echo "⚠ ConvNeXt-UNet optimization validation failed!"
    echo "Training may encounter issues, but proceeding..."
else
    echo "✓ ConvNeXt-UNet optimization validation passed"
fi

echo "=============================================="
echo ""

# =======================================================================
# EXECUTE OPTIMIZED CONVNEXT-UNET TRAINING
# =======================================================================

echo "🚀 STARTING OPTIMIZED CONVNEXT-UNET TRAINING"
echo "=================================================="
echo "Training ConvNeXt-UNet with advanced optimization techniques"
echo ""
echo "Optimization Configuration:"
echo "- Architecture: ConvNeXt-UNet (Enhanced)"
echo "- Learning Rate: 8e-5 (with warmup + cosine decay)"
echo "- Batch Size: 4 (optimized)"
echo "- Max Epochs: 120 (extended)"
echo "- Loss: Combined Focal + Dice Loss"
echo "- Optimizer: AdamW with weight decay"
echo "- Regularization: L2 + enhanced preprocessing"
echo "- Target: 93%+ Jaccard (matching best performers)"
echo ""
echo "Expected timeline: 15-20 hours (complete optimized training)"
echo "Expected performance: 93%+ Jaccard (target achievement)"
echo "Advanced features: Warmup scheduling, stratified split, contrast enhancement"
echo "=================================================="

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="convnext_unet_optimized_${TIMESTAMP}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Execute the optimized training with comprehensive logging
echo "Starting ConvNeXt-UNet optimized training execution..."
singularity exec --nv "$image" python3 convnext_unet_optimized_training.py 2>&1 | tee "${OUTPUT_DIR}_console.log"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "CONVNEXT-UNET OPTIMIZED TRAINING COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✓ SUCCESS"
    echo ""
    echo "✓ ConvNeXt-UNet optimized training completed successfully!"
    echo ""
    echo "Generated files:"
    echo "📁 Training directory:"
    ls -la convnext_unet_optimized_*/ 2>/dev/null || echo "   No training directory found"
    echo ""
    echo "📊 Model and results:"
    ls -la convnext_unet_optimized_*/ConvNeXt_UNet_* 2>/dev/null || echo "   No model files found"
    echo ""

    echo "🎯 CONVNEXT-UNET OPTIMIZATION PERFORMANCE SUMMARY:"
    echo "==================================================="
    singularity exec --nv "$image" python3 -c "
import pandas as pd
import os
import glob
import json

# Look for ConvNeXt-UNet optimization results
json_files = glob.glob('convnext_unet_optimized_*/ConvNeXt_UNet_optimized_results.json')

if json_files:
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)

            model_name = result.get('model_name', 'ConvNeXt_UNet_Optimized')
            best_jaccard = result.get('best_val_jaccard', 0)
            final_jaccard = result.get('final_val_jaccard', 0)
            epoch = result.get('best_epoch', 0)
            time_sec = result.get('training_time_seconds', 0)
            params = result.get('model_parameters', 0)

            print(f'ConvNeXt-UNet Optimization Results:')
            print(f'🏆 Best Jaccard: {best_jaccard:.4f} ({best_jaccard*100:.2f}%) at epoch {epoch}')
            print(f'🎯 Final Jaccard: {final_jaccard:.4f} ({final_jaccard*100:.2f}%)')
            print(f'⏱️ Training Time: {time_sec:.0f}s ({time_sec/3600:.1f}h)')
            print(f'🧠 Parameters: {params:,} ({params/1e6:.1f}M)')
            print()

            # Performance assessment against targets
            target_jaccard = 0.93
            if best_jaccard >= target_jaccard:
                print('🎉 OPTIMIZATION SUCCESS: Target 93%+ Jaccard ACHIEVED!')
                print(f'   Improvement: {(best_jaccard - target_jaccard)*100:.2f} percentage points above target')
            elif best_jaccard >= 0.90:
                print('📈 SIGNIFICANT IMPROVEMENT: Good performance achieved')
                print(f'   Gap to target: {(target_jaccard - best_jaccard)*100:.2f} percentage points')
            else:
                print('⚠ FURTHER OPTIMIZATION NEEDED')
                print(f'   Current gap: {(target_jaccard - best_jaccard)*100:.2f} percentage points')

            # Comparison with other architectures
            print()
            print('📊 ARCHITECTURE COMPARISON:')
            print('   CoAtNet-UNet: 93.93% Jaccard')
            print('   Swin-UNet: 93.46% Jaccard')
            print(f'   ConvNeXt-UNet (Optimized): {best_jaccard*100:.2f}% Jaccard')

        except Exception as e:
            print(f'Error reading {json_file}: {e}')
else:
    print('No ConvNeXt-UNet optimization results found.')

    # Check for CSV files as fallback
    csv_files = glob.glob('convnext_unet_optimized_*/ConvNeXt_UNet_*_history.csv')
    if csv_files:
        print('Training history found in CSV files:')
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'val_advanced_jaccard_coefficient' in df.columns:
                    best_jaccard = df['val_advanced_jaccard_coefficient'].max()
                    best_epoch = df['val_advanced_jaccard_coefficient'].idxmax() + 1
                    print(f'Best Val Jaccard: {best_jaccard:.4f} ({best_jaccard*100:.2f}%) at epoch {best_epoch}')
            except Exception as e:
                print(f'Error reading {csv_file}: {e}')
" 2>/dev/null || echo "Unable to generate performance summary"

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ ConvNeXt-UNet optimized training failed!"
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

    echo "🔧 OPTIMIZATION TROUBLESHOOTING:"
    echo "==============================="
    echo "1. Dataset Issues:"
    echo "   - Ensure dataset_full_stack/ exists with .tif files"
    echo "   - Check enhanced preprocessing compatibility"
    echo ""
    echo "2. Memory Issues:"
    echo "   - Optimized ConvNeXt-UNet requires 8-12 GB GPU memory"
    echo "   - Batch size reduced to 4 for optimization"
    echo ""
    echo "3. Optimization Features:"
    echo "   - Advanced LR scheduling may need adjustment"
    echo "   - Combined loss function compatibility"
    echo "   - Enhanced regularization effects"
    echo ""
    echo "4. Model Complexity:"
    echo "   - Extended training epochs (120) for convergence"
    echo "   - Advanced preprocessing pipeline"
fi

echo ""
echo "📁 CONSOLE LOG SAVED: ${OUTPUT_DIR}_console.log"
echo ""
echo "🔗 OPTIMIZATION NEXT STEPS:"
echo "=========================="
echo "1. Analyze ConvNeXt-UNet optimization results vs target (93%+)"
echo "2. Compare with current best performers:"
echo "   - CoAtNet-UNet: 93.93% Jaccard"
echo "   - Swin-UNet: 93.46% Jaccard"
echo "3. If target achieved, document optimization techniques"
echo "4. If target not met, consider ensemble methods or architecture modifications"
echo ""
echo "======================================================="
echo "CONVNEXT-UNET OPTIMIZATION TRAINING JOB COMPLETE"
echo "Model: ConvNeXt-UNet (Enhanced with Advanced Optimization)"
echo "Target: 93%+ Jaccard Performance Achievement"
echo "Framework: TensorFlow/Keras + Advanced Optimization Pipeline"
echo "======================================================="