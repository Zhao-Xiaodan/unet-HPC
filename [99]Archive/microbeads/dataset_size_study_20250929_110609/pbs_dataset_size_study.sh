#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Mitochondria_Dataset_Size_Study
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# MITOCHONDRIA SEGMENTATION - DATASET SIZE SUFFICIENCY STUDY
# =======================================================================
# Study how many images are sufficient for acceptable segmentation results
# Testing: 10%, 20%, 50%, 75%, 100% of full dataset
# Using FIXED Jaccard implementation for reliable metrics
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "MITOCHONDRIA SEGMENTATION - DATASET SIZE SUFFICIENCY STUDY"
echo "======================================================================="
echo "🔬 Study: Dataset size requirements for acceptable performance"
echo "📊 Testing: 10%, 20%, 50%, 75%, 100% of full dataset (1,980 patches)"
echo "✅ Using: FIXED Jaccard implementation"
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

echo "=== STUDY CONFIGURATION ==="
echo "🎯 Purpose: Determine minimum dataset size for acceptable results"
echo "📊 Full Dataset: ./dataset_full_stack/images/ (1980 patches)"
echo "📈 Test Percentages: 10%, 20%, 50%, 75%, 100%"
echo "🎯 Acceptable Threshold: Jaccard > 0.8"
echo "🏗️  Models: UNet, Attention UNet, Attention ResUNet"
echo "✅ Implementation: FIXED Jaccard coefficient"
echo "🔄 Training: Up to 100 epochs with early stopping"
echo "⏱️  Expected Total Time: ~20-24 hours for all percentages"
echo "=============================="
echo ""

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

echo "TensorFlow Container: $image"
echo ""

# =======================================================================
# PRE-EXECUTION CHECKS
# =======================================================================

echo "=== PRE-EXECUTION CHECKS ==="

# Check dataset structure
echo "1. Checking dataset structure..."
if [ -d "./dataset_full_stack/images/" ] && [ -d "./dataset_full_stack/masks/" ]; then
    echo "   ✓ Dataset directories found"

    # Count files
    img_count=$(find ./dataset_full_stack/images/ -name "*.tif" | wc -l)
    mask_count=$(find ./dataset_full_stack/masks/ -name "*.tif" | wc -l)

    echo "   ✓ Images found: $img_count .tif files"
    echo "   ✓ Masks found: $mask_count .tif files"

    if [ $img_count -eq 0 ] || [ $mask_count -eq 0 ]; then
        echo "   WARNING: No .tif files found in dataset directories"
        exit 1
    fi

    if [ $img_count -ne $mask_count ]; then
        echo "   WARNING: Unequal number of images ($img_count) and masks ($mask_count)"
        echo "   This may cause issues during training"
    fi
else
    echo "   ERROR: Dataset directories not found!"
    echo "   Expected: ./dataset_full_stack/images/ and ./dataset_full_stack/masks/"
    exit 1
fi

# Check if Python files exist
echo ""
echo "2. Checking Python files..."
if [ -f "./224_225_226_dataset_size_study.py" ] && [ -f "./224_225_226_models.py" ]; then
    echo "   ✓ Dataset size study script found"
    echo "   ✓ Models definition file found (with FIXED Jaccard)"
else
    echo "   ERROR: Required Python files not found"
    echo "   Expected:"
    echo "   - 224_225_226_dataset_size_study.py"
    echo "   - 224_225_226_models.py (with fixed Jaccard implementation)"
    ls -la ./*.py
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
print('GPU available:', tf.test.is_gpu_available())

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
    print('  ✗ focal_loss - May need installation')
    print('    Will attempt to install during execution')
"
echo "==============================="
echo ""

# =======================================================================
# CREATE DATASET SIZE STUDY WRAPPER
# =======================================================================

echo "=== PREPARING DATASET SIZE STUDY ==="
echo "Creating wrapper script with dependency handling..."

# Create a wrapper script that handles focal_loss installation and runs multiple dataset percentages
cat > run_dataset_size_study.py << 'EOF'
#!/usr/bin/env python3
"""
Wrapper script to run dataset size study with proper dependency handling.
"""

import os
import sys
import subprocess

def install_focal_loss():
    """Install focal_loss if not available"""
    try:
        from focal_loss import BinaryFocalLoss
        print("✓ focal_loss already available")
        return True
    except ImportError:
        print("Installing focal_loss...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "focal-loss"])
            print("✓ focal_loss installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install focal_loss: {e}")
            print("Creating custom focal_loss implementation...")
            create_custom_focal_loss()
            return True

def create_custom_focal_loss():
    """Create custom focal loss implementation if package not available"""
    with open('focal_loss.py', 'w') as f:
        f.write('''
import tensorflow as tf
from tensorflow.keras import backend as K

class BinaryFocalLoss:
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        """
        Binary focal loss implementation
        """
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Calculate focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)

        focal_loss = -alpha_t * K.pow(1 - pt, self.gamma) * K.log(pt)

        return K.mean(focal_loss)
''')

def run_dataset_percentage(percentage):
    """Run training for a specific dataset percentage"""
    print(f"\n{'='*60}")
    print(f"🔬 RUNNING DATASET SIZE STUDY: {percentage}% OF FULL DATASET")
    print(f"{'='*60}")

    # Set environment variable for dataset percentage
    os.environ['DATASET_PERCENTAGE'] = str(percentage)

    # Import and run the study script
    try:
        exec(open('224_225_226_dataset_size_study.py').read())
        print(f"✅ Completed study for {percentage}% dataset")
        return True
    except Exception as e:
        print(f"❌ Failed study for {percentage}% dataset: {e}")
        return False

def main():
    print("=== MITOCHONDRIA DATASET SIZE SUFFICIENCY STUDY ===")
    print("🔬 Testing multiple dataset sizes for acceptable performance")
    print()

    # Install focal loss
    install_focal_loss()

    # Dataset percentages to test
    percentages = [10, 20, 50, 75, 100]

    results_summary = {}

    print(f"\n🎯 DATASET SIZE STUDY PLAN:")
    print("-" * 30)
    for pct in percentages:
        estimated_samples = int(1980 * pct / 100)
        print(f"  {pct:3d}%: ~{estimated_samples:4d} samples")
    print()

    # Run study for each percentage
    successful_runs = 0
    for percentage in percentages:
        try:
            success = run_dataset_percentage(percentage)
            if success:
                successful_runs += 1
                results_summary[f'{percentage}%'] = 'Completed'
            else:
                results_summary[f'{percentage}%'] = 'Failed'
        except Exception as e:
            print(f"❌ Critical error for {percentage}%: {e}")
            results_summary[f'{percentage}%'] = f'Error: {e}'

    # Final summary
    print(f"\n{'='*60}")
    print("🏁 DATASET SIZE STUDY COMPLETED")
    print(f"{'='*60}")
    print(f"✅ Successful runs: {successful_runs}/{len(percentages)}")
    print("\n📋 RESULTS SUMMARY:")
    for pct, status in results_summary.items():
        print(f"  {pct:4s}: {status}")

    print(f"\n📁 Individual results saved in respective output directories")
    print("🔍 Check each directory for detailed analysis and visualizations")

if __name__ == "__main__":
    main()
EOF

echo "✓ Dataset size study wrapper script created"
echo "============================="
echo ""

# =======================================================================
# EXECUTE DATASET SIZE STUDY
# =======================================================================

echo "🔬 STARTING COMPREHENSIVE DATASET SIZE STUDY"
echo "=============================================="
echo "📊 Testing dataset percentages: 10%, 20%, 50%, 75%, 100%"
echo "🎯 Goal: Find minimum dataset size for acceptable performance (Jaccard > 0.8)"
echo "🏗️  Models: UNet, Attention UNet, Attention ResUNet"
echo "⏱️  Expected time: ~20-24 hours total"
echo "=============================================="

# Create main study output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_OUTPUT_DIR="dataset_size_study_${TIMESTAMP}"
mkdir -p "$MAIN_OUTPUT_DIR"

echo "📁 Main output directory: $MAIN_OUTPUT_DIR"
echo ""

# Execute the comprehensive study
singularity exec --nv "$image" python3 run_dataset_size_study.py 2>&1 | tee "${MAIN_OUTPUT_DIR}/study_console_${TIMESTAMP}.log"

# Capture exit code
EXIT_CODE=$?

# Move all generated study directories to main output
echo ""
echo "📦 Organizing study results..."
mv dataset_size_study_*pct_* "$MAIN_OUTPUT_DIR/" 2>/dev/null || echo "No individual study directories to move"

echo ""
echo "======================================================================="
echo "🔬 DATASET SIZE SUFFICIENCY STUDY COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✓ SUCCESS"
    echo ""
    echo "✅ Dataset size study completed successfully!"
    echo ""
    echo "📁 Study results organized in: $MAIN_OUTPUT_DIR"
    echo ""
    echo "📊 GENERATED ANALYSIS:"
    echo "====================="
    ls -la "$MAIN_OUTPUT_DIR"
    echo ""

    # Generate study summary if possible
    echo "🔍 STUDY SUMMARY:"
    echo "=================="
    singularity exec --nv "$image" python3 -c "
import os
import json
import glob

main_dir = '$MAIN_OUTPUT_DIR'
study_dirs = glob.glob(os.path.join(main_dir, 'dataset_size_study_*pct_*'))

print('📋 Dataset Size Study Results Summary:')
print('-' * 50)

results = []
for study_dir in sorted(study_dirs):
    # Extract percentage from directory name
    import re
    match = re.search(r'(\d+)pct', os.path.basename(study_dir))
    if match:
        percentage = int(match.group(1))

        # Look for results JSON
        json_files = glob.glob(os.path.join(study_dir, 'study_results_*.json'))
        if json_files:
            try:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)

                samples = data.get('used_samples', 0)
                models = data.get('models', {})

                # Calculate average Jaccard
                jaccards = [model_data['best_val_jaccard'] for model_data in models.values()]
                avg_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0

                # Count acceptable models (>0.8)
                acceptable = sum(1 for j in jaccards if j > 0.8)
                total = len(jaccards)

                print(f'{percentage:3d}% ({samples:4d} samples): Avg Jaccard = {avg_jaccard:.3f}, Acceptable = {acceptable}/{total}')
                results.append((percentage, samples, avg_jaccard, acceptable, total))

            except Exception as e:
                print(f'{percentage:3d}%: Error reading results - {e}')
        else:
            print(f'{percentage:3d}%: No results found')

print()
print('🎯 SUFFICIENCY ANALYSIS:')
print('-' * 25)
for pct, samples, avg_jaccard, acceptable, total in results:
    if acceptable == total:
        status = '✅ SUFFICIENT'
    elif acceptable > 0:
        status = '⚠️  PARTIAL'
    else:
        status = '❌ INSUFFICIENT'
    print(f'{pct:3d}% dataset: {status} (all models acceptable: {acceptable == total})')

if results:
    # Find minimum sufficient dataset
    sufficient_percentages = [pct for pct, samples, avg_jaccard, acceptable, total in results if acceptable == total]
    if sufficient_percentages:
        min_sufficient = min(sufficient_percentages)
        min_samples = next(samples for pct, samples, avg_jaccard, acceptable, total in results if pct == min_sufficient)
        print(f'')
        print(f'🏆 MINIMUM SUFFICIENT DATASET: {min_sufficient}% ({min_samples} samples)')
        print(f'💡 RECOMMENDATION: Use at least {min_sufficient}% of available data for reliable results')
    else:
        print(f'')
        print(f'⚠️  NO FULLY SUFFICIENT DATASET FOUND in tested percentages')
        print(f'💡 RECOMMENDATION: Consider using 100% of available data or collecting more samples')
" 2>/dev/null || echo "Unable to generate summary - check individual result files"

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Dataset size study failed!"
    echo ""
    echo "🔍 ERROR ANALYSIS:"
    echo "=================="
    echo "Check the console log for detailed error information:"
    echo "   ${MAIN_OUTPUT_DIR}/study_console_${TIMESTAMP}.log"
    echo ""

    if [ -f "${MAIN_OUTPUT_DIR}/study_console_${TIMESTAMP}.log" ]; then
        echo "Last 30 lines of console log:"
        tail -30 "${MAIN_OUTPUT_DIR}/study_console_${TIMESTAMP}.log"
    fi
fi

echo ""
echo "📁 ALL STUDY RESULTS SAVED IN: $MAIN_OUTPUT_DIR"
echo "📝 CONSOLE LOG: ${MAIN_OUTPUT_DIR}/study_console_${TIMESTAMP}.log"
echo ""
echo "======================================="
echo "🔬 DATASET SIZE SUFFICIENCY STUDY COMPLETE"
echo "Purpose: Determine minimum dataset requirements"
echo "Goal: Find dataset size for Jaccard > 0.8"
echo "======================================="