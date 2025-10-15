
#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Train_UNets_512_Xukuang
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# TRAINING JOB: U-NET, ATTENTION U-NET, ATTENTION RES-UNET
# (Xukuang's Parameters)
# =======================================================================
# This script runs a training job for three U-Net architectures using
# a fixed set of hyperparameters on 512x512 grayscale images.
# =======================================================================

cd /home/svu/phyzxi/scratch/unet-HPC

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONSOLE_LOG="training_console_xukuang_${TIMESTAMP}.log"

echo "======================================================================="
echo "MODEL TRAINING - 512x512 IMAGES (Xukuang's Parameters)"
echo "======================================================================="
echo "Architectures: U-Net, Attention U-Net, Attention Res-UNet"
echo "Dataset: dataset_shrunk_masks"
echo "Image Size: 512x512 GRAYSCALE"

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "This output is being logged to: $CONSOLE_LOG"
echo ""

# =======================================================================
# HYPERPARAMETER CONFIGURATION
# =======================================================================
echo "=== FIXED HYPERPARAMETERS ==="
echo "Source: bead_seg.ipynb"
echo "  - Learning Rate: 5e-3"
echo "  - Batch Size: 4"
echo "  - Epochs: 200"
echo "  - Optimizer: Adam"
echo "  - Loss Function: BinaryFocalLoss(gamma=2)"
echo "  - Base Filters: 64"
echo "  - Dropout: 0.0"
echo "============================="
echo ""

# =======================================================================
# ENVIRONMENT SETUP
# =======================================================================
echo "=== ENVIRONMENT SETUP ==="

export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

module load singularity

image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

if [ ! -f "$image" ]; then
    echo "ERROR: TensorFlow container not found at $image"
    exit 1
fi

echo "✓ TensorFlow Container: $image"
echo "✓ GPU memory growth: ENABLED"
echo "==========================="
echo ""

# =======================================================================
# PRE-EXECUTION CHECKS
# =======================================================================
echo "=== PRE-EXECUTION CHECKS ==="

REQUIRED_FILES=("train_models_512_xukuang_parameters.py" "models.py" "loss_functions_fixed.py")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "./$file" ]; then
        echo "   ✗ ERROR: Required script '$file' not found!"
        exit 1
    fi
    echo "   ✓ Found required script: $file"
done

echo ""
echo "2. Checking dataset..."
if [ ! -d "./dataset_shrunk_masks/images/" ] || [ ! -d "./dataset_shrunk_masks/masks/" ]; then
    echo "   ✗ ERROR: dataset_shrunk_masks not found!"
    exit 1
fi

img_count=$(find ./dataset_shrunk_masks/images/ -name "*.png" 2>/dev/null | wc -l)
mask_count=$(find ./dataset_shrunk_masks/masks/ -name "*.png" 2>/dev/null | wc -l)

echo "   ✓ Images: $img_count .png files"
echo "   ✓ Masks: $mask_count .png files"

if [ $img_count -eq 0 ] || [ $mask_count -eq 0 ]; then
    echo "   ✗ ERROR: No images or masks found in dataset"
    exit 1
fi
echo "============================="
echo ""

# =======================================================================
# EXECUTE TRAINING SCRIPT
# =======================================================================
echo "🚀 STARTING MODEL TRAINING (Xukuang's Parameters)"
echo "============================================="
echo "Training 3 models: U-Net, Attention U-Net, Attention Res-UNet"
echo "Progress will be logged to the console and saved to $CONSOLE_LOG"
echo "============================================="
echo ""

singularity exec --nv "$image" python3 train_models_512_xukuang_parameters.py 2>&1 | tee "$CONSOLE_LOG"

EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "TRAINING JOB COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✓ SUCCESS"
    echo ""
    echo "✓ Training script completed successfully!"
    echo ""

    # Find the output directory created by the Python script (the most recent one)
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "training_run_xukuang_*" -mmin -10 | sort | tail -1)

    if [ -d "$OUTPUT_DIR" ]; then
        echo "📁 RESULTS DIRECTORY: $OUTPUT_DIR"
        echo ""

        # Archive logs and scripts for reproducibility
        echo "📋 Archiving logs and scripts..."
        cp "$CONSOLE_LOG" "$OUTPUT_DIR/"
        echo "   ✓ Copied console log: $CONSOLE_LOG -> $OUTPUT_DIR/"
        
        if [ -n "$PBS_JOBID" ]; then
            PBS_OUTPUT_FILE="${PBS_JOBNAME}.o${PBS_JOBID}"
            if [ -f "$PBS_OUTPUT_FILE" ]; then
                cp "$PBS_OUTPUT_FILE" "$OUTPUT_DIR/"
                echo "   ✓ Copied PBS output file: $PBS_OUTPUT_FILE -> $OUTPUT_DIR/"
            fi
        fi

        cp train_models_512_xukuang_parameters.py "$OUTPUT_DIR/"
        cp pbs_train_models_512_xukuang_parameters.sh "$OUTPUT_DIR/"
        cp models.py "$OUTPUT_DIR/"
        cp loss_functions_fixed.py "$OUTPUT_DIR/"
        echo "   ✓ Copied source scripts to $OUTPUT_DIR/ for reproducibility."
        echo ""

        echo "📊 FINAL SUMMARY:"
        if [ -f "$OUTPUT_DIR/summary.md" ]; then
            cat "$OUTPUT_DIR/summary.md"
        else
            echo "   (Summary file not found)"
        fi

    else
        echo "⚠ WARNING: Could not find the output directory created by the training script."
    fi

else
    echo "Exit code: $EXIT_CODE ✗ ERROR"
    echo ""
    echo "✗ Training script failed!"
    echo ""

    echo "Last 50 lines of console log ($CONSOLE_LOG):"
    echo "-----------------------------"
    tail -50 "$CONSOLE_LOG"
    echo ""
    
    echo "🔧 COMMON ISSUES:"
    echo "================="
    echo "1. Out of Memory (OOM): Check the log for 'ResourceExhaustedError'."
    echo "   - The models are using 64 base filters, which is memory-intensive."
    echo "   - To fix, you can edit 'models.py' and reduce 'FILTER_NUM' from 64 to 32 or 16."
    echo ""
    echo "2. Missing Files: Ensure all required .py scripts are in the same directory."
fi

echo ""
echo "======================================="
