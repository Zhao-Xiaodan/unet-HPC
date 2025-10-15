#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Modern_UNet_Mitochondria_Segmentation
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# MODERN U-NET ARCHITECTURES FOR MITOCHONDRIA SEGMENTATION - PBS SCRIPT
# =======================================================================
# Training ConvNeXt-UNet, Swin-UNet, and CoAtNet-UNet for mitochondria segmentation
# State-of-the-art architectures with improved feature extraction capabilities
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "MODERN U-NET ARCHITECTURES - MITOCHONDRIA SEGMENTATION TRAINING"
echo "======================================================================="
echo "Models: ConvNeXt-UNet, Swin-UNet, CoAtNet-UNet"
echo "Task: Mitochondria semantic segmentation with state-of-the-art architectures"
echo "Framework: TensorFlow/Keras"
echo "Expected Training Time: 24-48 hours (due to model complexity)"
echo ""

# Job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $(free -h | grep Mem | awk '{print $2}'), CPUs: $(nproc)"
echo ""

# =======================================================================
# ENVIRONMENT SETUP
# =======================================================================

echo "=== TRAINING CONFIGURATION ==="
echo "Dataset Images: ./dataset_full_stack/images/ (1980 patches - REQUIRED)"
echo "Dataset Masks: ./dataset_full_stack/masks/ (1980 patches - REQUIRED)"
echo "Alternative: ./dataset/images/ and ./dataset/masks/"
echo "Image Size: 256x256x3"
echo "Batch Size: 4 (smaller due to model complexity)"
echo "Learning Rate: 1e-4 (conservative for modern architectures)"
echo "Epochs per Model: 100 (with early stopping)"
echo "Models: 3 (ConvNeXt-UNet, Swin-UNet, CoAtNet-UNet)"
echo "Loss Function: Binary Focal Loss"
echo "Optimizers: AdamW for transformers, Adam for ConvNeXt"
echo "=============================="
echo ""

# TensorFlow environment variables
export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export CUDA_VISIBLE_DEVICES=0

# Additional memory optimizations for complex models
export TF_GPU_MEMORY_ALLOW_GROWTH=true
export TF_FORCE_GPU_ALLOW_GROWTH=true

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

echo "TensorFlow Container: $image"
echo ""

# =======================================================================
# PRE-EXECUTION CHECKS
# =======================================================================

echo "=== PRE-EXECUTION CHECKS ==="

# Check dataset structure (prioritize full stack dataset)
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
    echo "   Expected one of:"
    echo "   - ./dataset_full_stack/images/ and ./dataset_full_stack/masks/ (PREFERRED)"
    echo "   - ./dataset/images/ and ./dataset/masks/"
    echo "   Current directory contents:"
    ls -la ./
    echo ""
    echo "   🔧 SOLUTION: Ensure your dataset is properly structured."
    echo "   The full stack dataset (1980 patches) is strongly recommended for modern architectures."
    exit 1
fi

if [ $img_count -eq 0 ] || [ $mask_count -eq 0 ]; then
    echo "   WARNING: No image files found in dataset directories"
    echo "   Please ensure your dataset contains .tif, .png, or .jpg files"
fi

if [ $img_count -ne $mask_count ]; then
    echo "   WARNING: Unequal number of images ($img_count) and masks ($mask_count)"
    echo "   This may cause issues during training"
fi

# Recommend dataset size for modern architectures
if [ $img_count -lt 1000 ]; then
    echo "   WARNING: Small dataset detected ($img_count images)"
    echo "   Modern architectures perform better with larger datasets (1000+ samples)"
    echo "   Consider using the full stack dataset for optimal results"
fi

# Check if Python files exist
echo ""
echo "2. Checking Python files..."
required_files=("modern_unet_models.py" "modern_unet_training.py")
missing_files=()

for file in "${required_files[@]}"; do
    if [ -f "./$file" ]; then
        echo "   ✓ $file found"
    else
        echo "   ✗ $file NOT found"
        missing_files+=("$file")
    fi
done

# Check for original models.py (dependency)
if [ -f "./224_225_226_models.py" ] && [ ! -f "./models.py" ]; then
    echo "   ✓ Renaming 224_225_226_models.py to models.py"
    cp ./224_225_226_models.py ./models.py
fi

if [ ! -f "./models.py" ]; then
    echo "   ✗ models.py NOT found (required dependency)"
    missing_files+=("models.py")
fi

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "   ERROR: Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "   Please ensure all required Python files are present before running."
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
import os

print('Python version:', sys.version.split()[0])
print('TensorFlow version:', tf.__version__)
print('CUDA built support:', tf.test.is_built_with_cuda())

# List GPUs
gpus = tf.config.list_physical_devices('GPU')
print('Physical GPUs found:', len(gpus))
for i, gpu in enumerate(gpus):
    print(f'  GPU {i}: {gpu}')

# Enable memory growth (critical for modern architectures)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('✓ GPU memory growth enabled (CRITICAL for modern U-Nets)')
    except RuntimeError as e:
        print('Memory growth error:', e)

# Test GPU operation
if gpus:
    try:
        with tf.device('/GPU:0'):
            # Test larger tensor operation (modern models need more memory)
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
        print('✓ GPU operation test successful')

        # Check GPU memory
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        if 'device_name' in gpu_details:
            print(f'✓ GPU Details: {gpu_details[\"device_name\"]}')
    except Exception as e:
        print('GPU operation failed:', e)
        print('WARNING: This may cause issues with modern U-Net training')

# Check key dependencies
deps_to_check = ['cv2', 'PIL', 'matplotlib', 'numpy', 'sklearn', 'pandas']
print()
print('Checking dependencies:')
for dep in deps_to_check:
    try:
        __import__(dep)
        print(f'  ✓ {dep}')
    except ImportError:
        print(f'  ✗ {dep} - Missing!')

# Special check for focal_loss
try:
    from focal_loss import BinaryFocalLoss
    print('  ✓ focal_loss')
except ImportError:
    print('  ⚠ focal_loss - Will create custom implementation')

print()
print('Memory recommendations for modern U-Nets:')
print('- ConvNeXt-UNet: ~8-12 GB GPU memory')
print('- Swin-UNet: ~10-16 GB GPU memory')
print('- CoAtNet-UNet: ~6-10 GB GPU memory')
print('- Batch size will be automatically adjusted if needed')
"
echo "==============================="
echo ""

# =======================================================================
# MODEL VALIDATION TEST
# =======================================================================

echo "=== MODEL VALIDATION TEST ==="
echo "Testing model creation to validate implementation..."

singularity exec --nv "$image" python3 -c "
try:
    import sys
    sys.path.append('.')

    print('Testing modern U-Net model creation...')
    from modern_unet_models import create_modern_unet

    # Test all models with minimal input
    input_shape = (64, 64, 3)  # Small test shape
    models_to_test = ['ConvNeXt_UNet', 'Swin_UNet', 'CoAtNet_UNet']

    for model_name in models_to_test:
        try:
            model = create_modern_unet(model_name, input_shape, num_classes=1)
            params = model.count_params()
            print(f'  ✓ {model_name}: {params:,} parameters')
            del model  # Free memory
        except Exception as e:
            print(f'  ✗ {model_name}: Failed - {e}')

    print('✓ Model validation completed')

except Exception as e:
    print(f'✗ Model validation failed: {e}')
    print('This indicates issues with the model implementation.')
    import traceback
    traceback.print_exc()
"

model_test_exit_code=$?
if [ $model_test_exit_code -ne 0 ]; then
    echo "⚠ Model validation test failed!"
    echo "Training may encounter issues, but proceeding..."
else
    echo "✓ Model validation test passed"
fi

echo "========================="
echo ""

# =======================================================================
# EXECUTE TRAINING
# =======================================================================

echo "🚀 STARTING MODERN U-NET TRAINING"
echo "=============================================="
echo "Training 3 state-of-the-art models sequentially:"
echo "1. ConvNeXt-UNet (ConvNeXt blocks)"
echo "2. Swin-UNet (Swin Transformer blocks)"
echo "3. CoAtNet-UNet (Convolution + Attention)"
echo ""
echo "Training Configuration:"
echo "- Learning Rate: 1e-4 (conservative for stability)"
echo "- Batch Size: 4 (adjusted for model complexity)"
echo "- Max Epochs: 100 per model (with early stopping)"
echo "- Optimizers: AdamW for transformers, Adam for ConvNeXt"
echo "- Loss: Binary Focal Loss (gamma=2)"
echo ""
echo "Expected total time: 24-48 hours"
echo "Results will be saved as .hdf5 model files and training logs"
echo "=============================================="

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="modern_unet_training_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Clear any existing TensorFlow caches before training
echo "Clearing TensorFlow caches to prevent dataset conflicts..."
rm -rf ~/.tensorflow_datasets /tmp/tf_data_cache /tmp/tensorflow_cache /tmp/tfds 2>/dev/null || true

# Execute the training with enhanced error handling and logging
echo "Starting training execution..."
echo "Note: Enhanced fixes applied for ConvNeXt-UNet dataset caching and CoAtNet-UNet weight initialization"
singularity exec --nv "$image" python3 modern_unet_training.py 2>&1 | tee "${OUTPUT_DIR}/training_console_${TIMESTAMP}.log"

# Capture exit code
EXIT_CODE=$?

# Move generated files to output directory
echo ""
echo "Moving generated files to output directory..."
mv modern_unet_training_*/* "$OUTPUT_DIR/" 2>/dev/null || echo "No training output directories to move"
mv *.hdf5 "$OUTPUT_DIR/" 2>/dev/null || echo "No .hdf5 files to move"
mv *.csv "$OUTPUT_DIR/" 2>/dev/null || echo "No .csv files to move"
mv *.json "$OUTPUT_DIR/" 2>/dev/null || echo "No .json files to move"
mv *.png "$OUTPUT_DIR/" 2>/dev/null || echo "No .png files to move"
mv custom_focal_loss.py "$OUTPUT_DIR/" 2>/dev/null || echo "No custom focal loss file to move"

echo ""
echo "======================================================================="
echo "MODERN U-NET TRAINING COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✓ SUCCESS"
    echo ""
    echo "✓ Modern U-Net training completed successfully!"
    echo ""
    echo "Generated files in $OUTPUT_DIR:"
    echo "📁 Model files (.hdf5):"
    ls -la "$OUTPUT_DIR"/*.hdf5 2>/dev/null || echo "   No model files found"
    echo ""
    echo "📊 Training histories (.csv):"
    ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "   No history files found"
    echo ""
    echo "📈 Training results (.json):"
    ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "   No result files found"
    echo ""
    echo "🖼️ Visualization files (.png):"
    ls -la "$OUTPUT_DIR"/*.png 2>/dev/null || echo "   No plot files found"
    echo ""

    echo "🎯 MODERN U-NET PERFORMANCE SUMMARY:"
    echo "======================================"
    singularity exec --nv "$image" python3 -c "
import pandas as pd
import os
import glob
import json

output_dir = '$OUTPUT_DIR'

# Look for results JSON files
json_files = glob.glob(os.path.join(output_dir, '*_results.json'))

if json_files:
    print('Modern U-Net Training Results:')
    print('-' * 50)

    results = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            results.append(data)
        except Exception as e:
            print(f'Error reading {json_file}: {e}')

    if results:
        # Sort by performance
        results.sort(key=lambda x: x.get('best_val_jaccard', 0), reverse=True)

        print()
        for i, result in enumerate(results, 1):
            model_name = result.get('model_name', 'Unknown')
            jaccard = result.get('best_val_jaccard', 0)
            epoch = result.get('best_epoch', 0)
            time_sec = result.get('training_time_seconds', 0)
            params = result.get('model_parameters', 0)
            stability = result.get('val_loss_stability', 0)

            print(f'{i}. {model_name}:')
            print(f'   🏆 Best Jaccard: {jaccard:.4f} (epoch {epoch})')
            print(f'   ⏱️ Training Time: {time_sec:.0f}s ({time_sec/3600:.1f}h)')
            print(f'   🧠 Parameters: {params:,} ({params/1e6:.1f}M)')
            print(f'   📊 Stability: {stability:.4f}')
            print()

        # Find best model
        best = max(results, key=lambda x: x.get('best_val_jaccard', 0))
        print(f'🥇 BEST PERFORMING MODEL: {best.get(\"model_name\", \"Unknown\")}')
        print(f'   Jaccard: {best.get(\"best_val_jaccard\", 0):.4f}')
        print(f'   Time: {best.get(\"training_time_seconds\", 0)/3600:.1f}h')
        print()

        # Performance comparison
        if len(results) > 1:
            print('📈 PERFORMANCE COMPARISON:')
            jaccards = [r.get('best_val_jaccard', 0) for r in results]
            best_jaccard = max(jaccards)
            print(f'   Best: {best_jaccard:.4f}')
            print(f'   Mean: {sum(jaccards)/len(jaccards):.4f}')
            print(f'   Range: {min(jaccards):.4f} - {max(jaccards):.4f}')
    else:
        print('No valid results found in JSON files.')
else:
    print('No training results JSON files found.')

    # Fallback: check CSV files
    csv_files = glob.glob(os.path.join(output_dir, '*history*.csv'))
    if csv_files:
        print()
        print('Training History Summary (from CSV files):')
        print('-' * 40)

        for csv_file in csv_files:
            model_name = os.path.basename(csv_file).split('_')[0]
            try:
                df = pd.read_csv(csv_file)
                if 'val_jacard_coef' in df.columns:
                    best_val_jaccard = df['val_jacard_coef'].max()
                    best_epoch = df['val_jacard_coef'].idxmax() + 1
                    final_loss = df['val_loss'].iloc[-1]
                    print(f'{model_name}:')
                    print(f'  Best Val Jaccard: {best_val_jaccard:.4f} (epoch {best_epoch})')
                    print(f'  Final Val Loss: {final_loss:.4f}')
                    print()
            except Exception as e:
                print(f'Error reading {csv_file}: {e}')
" 2>/dev/null || echo "Unable to generate performance summary"

    echo ""
    echo "🔬 ARCHITECTURE INSIGHTS:"
    echo "========================="
    echo "ConvNeXt-UNet: Modern CNN with improved efficiency"
    echo "Swin-UNet: Hierarchical Vision Transformer with shifted windows"
    echo "CoAtNet-UNet: Hybrid convolution + attention mechanism"
    echo ""
    echo "For detailed analysis, check:"
    echo "- Training curves: modern_unet_training_comparison.png"
    echo "- Performance summary: modern_unet_performance_summary.png"
    echo "- Individual model results: *_results.json files"

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Modern U-Net training failed!"
    echo ""
    echo "🔍 ERROR ANALYSIS:"
    echo "=================="
    echo "Check the console log for detailed error information:"
    echo "   ${OUTPUT_DIR}/training_console_${TIMESTAMP}.log"
    echo ""

    if [ -f "${OUTPUT_DIR}/training_console_${TIMESTAMP}.log" ]; then
        echo "Last 30 lines of console log:"
        tail -30 "${OUTPUT_DIR}/training_console_${TIMESTAMP}.log"
    fi
    echo ""

    echo "🔧 COMMON ISSUES AND SOLUTIONS:"
    echo "==============================="
    echo "1. GPU Memory Issues:"
    echo "   - Modern U-Nets require significant GPU memory"
    echo "   - Try reducing batch_size to 2 in modern_unet_training.py"
    echo "   - Ensure GPU memory growth is enabled"
    echo ""
    echo "2. Dataset Issues:"
    echo "   - Ensure dataset_full_stack/ exists with .tif files"
    echo "   - Check image/mask count matching"
    echo "   - Verify file permissions"
    echo ""
    echo "3. Dependency Issues:"
    echo "   - focal_loss package installation"
    echo "   - TensorFlow version compatibility"
    echo "   - Custom layer implementations"
    echo ""
    echo "4. Model Complexity:"
    echo "   - Modern architectures are computationally intensive"
    echo "   - Consider training individual models separately"
    echo "   - Check available GPU memory vs. model requirements"
    echo ""
    echo "5. Memory Optimization:"
    echo "   - Set export TF_GPU_MEMORY_ALLOW_GROWTH=true"
    echo "   - Use mixed precision training if needed"
    echo "   - Consider gradient checkpointing for large models"
fi

echo ""
echo "📁 ALL OUTPUT FILES SAVED IN: $OUTPUT_DIR"
echo "📝 CONSOLE LOG: ${OUTPUT_DIR}/training_console_${TIMESTAMP}.log"
echo ""
echo "🔗 NEXT STEPS:"
echo "============="
echo "1. Analyze training results and model performance"
echo "2. Compare with traditional U-Net architectures"
echo "3. Consider ensemble methods for improved performance"
echo "4. Optimize best-performing architecture for deployment"
echo ""
echo "========================================="
echo "MODERN U-NET TRAINING JOB COMPLETE"
echo "Models: ConvNeXt-UNet, Swin-UNet, CoAtNet-UNet"
echo "Framework: TensorFlow/Keras + Modern Architectures"
echo "========================================="