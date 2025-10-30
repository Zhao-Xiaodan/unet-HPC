#!/usr/bin/bash
#PBS -N Layer1_Kernels
#PBS -l walltime=0:30:00
#PBS -l select=1:ncpus=4:mpiprocs=1:ompthreads=4:mem=16gb
#PBS -j oe
#PBS -k oed
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

echo "========================================"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "========================================"

echo ""
echo "Configuration:"
echo "  Purpose: Visualize Layer 1 conv1 kernels (AlexNet Figure 3 style)"
echo "  Models:"
echo "    1. Frozen Gabor (should match initial Gabor filters)"
echo "    2. Trainable Gabor (should show minimal adaptation)"
echo "    3. Baseline U-Net (random initialization, learned filters)"
echo "  Output: layer1_kernel_visualizations/"
echo ""

echo "========================================"
echo "Setting up environment..."
echo "========================================"

cd $PBS_O_WORKDIR || exit 1

# Activate conda environment
source ~/.bashrc
conda activate unetCNN

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Create output directory
OUTPUT_DIR="./layer1_kernel_visualizations"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "1. Frozen Gabor Layer 1 Kernels"
echo "========================================"

FROZEN_MODEL="./edge_detector_experiment/20251030_122713/unet_frozen_layer1/best_model.pth"
FROZEN_OUTPUT="$OUTPUT_DIR/frozen_gabor_layer1_kernels.png"

if [ -f "$FROZEN_MODEL" ]; then
    echo "Model: $FROZEN_MODEL"
    echo "Output: $FROZEN_OUTPUT"

    python visualize_layer1_kernels.py \
        --model_path "$FROZEN_MODEL" \
        --output "$FROZEN_OUTPUT" \
        --title "Layer 1 Conv1 Kernels: Frozen Gabor (Should Be Identical to Initial)" \
        --n_filters 32 \
        --dropout 0.2

    echo "✓ Frozen Gabor kernels visualized"
else
    echo "❌ Model not found: $FROZEN_MODEL"
fi

echo ""
echo "========================================"
echo "2. Trainable Gabor Layer 1 Kernels"
echo "========================================"

TRAINABLE_MODEL="./edge_detector_experiment/20251030_122737/unet_trainable_layer1/best_model.pth"
TRAINABLE_OUTPUT="$OUTPUT_DIR/trainable_gabor_layer1_kernels.png"

if [ -f "$TRAINABLE_MODEL" ]; then
    echo "Model: $TRAINABLE_MODEL"
    echo "Output: $TRAINABLE_OUTPUT"

    python visualize_layer1_kernels.py \
        --model_path "$TRAINABLE_MODEL" \
        --output "$TRAINABLE_OUTPUT" \
        --title "Layer 1 Conv1 Kernels: Trainable Gabor (After 48 Epochs)" \
        --n_filters 32 \
        --dropout 0.2

    echo "✓ Trainable Gabor kernels visualized"
else
    echo "❌ Model not found: $TRAINABLE_MODEL"
fi

echo ""
echo "========================================"
echo "3. Baseline U-Net Layer 1 Kernels"
echo "========================================"

BASELINE_MODEL="./best_models_PyTorch/unet/best_model.pth"
BASELINE_OUTPUT="$OUTPUT_DIR/baseline_unet_layer1_kernels.png"

if [ -f "$BASELINE_MODEL" ]; then
    echo "Model: $BASELINE_MODEL"
    echo "Output: $BASELINE_OUTPUT"

    python visualize_layer1_kernels.py \
        --model_path "$BASELINE_MODEL" \
        --output "$BASELINE_OUTPUT" \
        --title "Layer 1 Conv1 Kernels: Baseline U-Net (Random Initialization)" \
        --n_filters 32 \
        --dropout 0.2

    echo "✓ Baseline U-Net kernels visualized"
else
    echo "❌ Model not found: $BASELINE_MODEL"
fi

echo ""
echo "========================================"
echo "4. Copy Initial Gabor Filters for Comparison"
echo "========================================"

INITIAL_GABOR="./edge_detector_experiment/20251030_122713/unet_frozen_layer1/gabor_filters_initial.png"
INITIAL_COPY="$OUTPUT_DIR/initial_gabor_filters_reference.png"

if [ -f "$INITIAL_GABOR" ]; then
    cp "$INITIAL_GABOR" "$INITIAL_COPY"
    echo "✓ Copied initial Gabor filters to: $INITIAL_COPY"
    echo "  (For visual comparison with frozen model)"
else
    echo "⚠ Initial Gabor visualization not found: $INITIAL_GABOR"
fi

echo ""
echo "========================================"
echo "Visualization complete!"
echo "End Time: $(date)"
echo "========================================"

echo ""
echo "Output directory structure:"
ls -lh "$OUTPUT_DIR/"

echo ""
echo "========================================"
echo "✓ Job complete!"
echo "========================================"

echo ""
echo "Expected results:"
echo "  1. frozen_gabor_layer1_kernels.png"
echo "     → Should be IDENTICAL to initial_gabor_filters_reference.png"
echo "     → Confirms freezing mechanism worked correctly"
echo ""
echo "  2. trainable_gabor_layer1_kernels.png"
echo "     → Should be VISUALLY SIMILAR to initial Gabor filters"
echo "     → May show subtle adaptations (1.04% mean change expected)"
echo ""
echo "  3. baseline_unet_layer1_kernels.png"
echo "     → Should show RANDOM-LOOKING patterns"
echo "     → May include textures, blobs, unstructured features"
echo "     → Contrast with structured Gabor edge detectors"
echo ""
echo "Scientific question:"
echo "  Do trainable Gabor filters maintain edge-detection character"
echo "  or drift toward texture-based features like baseline?"
echo ""
