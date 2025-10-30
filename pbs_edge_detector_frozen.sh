#!/usr/bin/bash
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -N EdgeDet_Frozen
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# Edge Detector Experiment: Frozen Layer 1
# ==============================================================================
#
# Controlled experiment to test whether Gabor filter initialization helps U-Net
# performance for cell counting.
#
# This script: FROZEN layer 1
# - Initialize enc1.conv1 with Gabor filters (edge detectors)
# - Freeze enc1 throughout training (edge detectors stay fixed)
# - Train remaining layers for 100 epochs
#
# Expected outcome:
# - If IoU ≈ baseline (0.6377): Edge detectors are sufficient
# - If IoU < baseline: Texture features are better (current observation justified)
# - If IoU > baseline: Edge detectors help (random init missed them)
#
# Expected runtime: ~20-24 hours (100 epochs, batch_size=4)
# ==============================================================================

# Load modules
module load singularity

# Singularity container
image=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif

# Change to working directory
cd $PBS_O_WORKDIR

echo "========================================"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "========================================"
echo ""

# Configuration
OUTPUT_DIR="./edge_detector_experiment"

echo "Experiment: Edge Detector Transfer Learning"
echo "  Variant: FROZEN layer 1"
echo "  Strategy: Gabor filters in enc1.conv1 stay fixed"
echo "  Output directory: $OUTPUT_DIR"
echo "  Expected runtime: ~20-24 hours"
echo ""

echo "Hyperparameters (from best model):"
echo "  n_filters: 32"
echo "  dropout: 0.2"
echo "  learning_rate: 0.001"
echo "  epochs: 100 (with early stopping)"
echo "  batch_size: 4"
echo ""

echo "Baseline for comparison:"
echo "  Model: Standard U-Net (random init)"
echo "  Val IoU: 0.6377"
echo ""

echo "========================================"
echo "Starting training..."
echo "========================================"
echo ""

# Run training with frozen layer 1
singularity exec --nv $image python train_edge_detector_experiment.py \
    --freeze_layer1 \
    --output_dir $OUTPUT_DIR

echo ""
echo "========================================"
echo "Training complete!"
echo "End Time: $(date)"
echo "========================================"
echo ""

# Show results
echo "Results summary:"
if [ -f "$OUTPUT_DIR"/*/unet_frozen_layer1/model_info.json ]; then
    echo ""
    echo "Frozen Layer 1 Results:"
    cat "$OUTPUT_DIR"/*/unet_frozen_layer1/model_info.json
    echo ""
fi

echo "Next steps:"
echo "1. Check training history:"
echo "   cat $OUTPUT_DIR/*/unet_frozen_layer1/training_history.csv"
echo ""
echo "2. Compare layer 1 weights:"
echo "   # Initial vs Final weights to verify frozen"
echo "   python -c \"import torch; w0=torch.load('$OUTPUT_DIR/*/unet_frozen_layer1/layer1_weights_epoch000.pth'); wf=torch.load('$OUTPUT_DIR/*/unet_frozen_layer1/layer1_weights_final.pth'); print('Weights changed:', not torch.equal(w0, wf))\""
echo ""
echo "3. Run feature visualization:"
echo "   python unet_feature_visualization.py \\"
echo "       --model_path $OUTPUT_DIR/*/unet_frozen_layer1/best_model.pth \\"
echo "       --n_filters 32 \\"
echo "       --dropout 0.2"
echo ""
echo "4. Compare with trainable layer 1 results"
echo ""
