#!/usr/bin/bash
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -N EdgeDet_Train
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# Edge Detector Experiment: Trainable Layer 1
# ==============================================================================
#
# Controlled experiment to test whether Gabor filter initialization helps U-Net
# performance for cell counting.
#
# This script: TRAINABLE layer 1
# - Initialize enc1.conv1 with Gabor filters (edge detectors)
# - Allow enc1 to train normally (edge detectors can adapt)
# - Train all layers for 100 epochs
#
# Expected outcome:
# - If weights stay edge-like: Edge detectors are optimal
# - If weights evolve to textures: Task prefers texture features
# - IoU comparison: Does Gabor init help vs random init?
#
# Key analysis: Compare layer1_weights_epoch000.pth vs layer1_weights_final.pth
# to see if/how Gabor filters adapt during training
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
echo "  Variant: TRAINABLE layer 1"
echo "  Strategy: Gabor filters in enc1.conv1 can adapt"
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

echo "Key Questions:"
echo "  1. Do Gabor filters provide better initialization than random?"
echo "  2. Do Gabor filters adapt toward texture features?"
echo "  3. Is the absence of edge detectors (baseline) a bug or feature?"
echo ""

echo "========================================"
echo "Starting training..."
echo "========================================"
echo ""

# Run training WITHOUT freezing layer 1
singularity exec --nv $image python train_edge_detector_experiment.py \
    --output_dir $OUTPUT_DIR

echo ""
echo "========================================"
echo "Training complete!"
echo "End Time: $(date)"
echo "========================================"
echo ""

# Show results
echo "Results summary:"
if [ -f "$OUTPUT_DIR"/*/unet_trainable_layer1/model_info.json ]; then
    echo ""
    echo "Trainable Layer 1 Results:"
    cat "$OUTPUT_DIR"/*/unet_trainable_layer1/model_info.json
    echo ""
fi

echo "Next steps:"
echo "1. Check training history:"
echo "   cat $OUTPUT_DIR/*/unet_trainable_layer1/training_history.csv"
echo ""
echo "2. Analyze layer 1 weight evolution:"
echo "   python -c \"import torch; w0=torch.load('$OUTPUT_DIR/*/unet_trainable_layer1/layer1_weights_epoch000.pth'); wf=torch.load('$OUTPUT_DIR/*/unet_trainable_layer1/layer1_weights_final.pth'); print('Weights changed:', not torch.equal(w0, wf)); print('L2 distance:', torch.norm(wf-w0).item())\""
echo ""
echo "3. Run feature visualization:"
echo "   python unet_feature_visualization.py \\"
echo "       --model_path $OUTPUT_DIR/*/unet_trainable_layer1/best_model.pth \\"
echo "       --n_filters 32 \\"
echo "       --dropout 0.2"
echo ""
echo "4. Compare with frozen layer 1 and baseline results"
echo ""
echo "Expected outcomes:"
echo "  - If layer 1 weights stay similar to Gabor: Edge detectors are good"
echo "  - If layer 1 weights diverge significantly: Textures preferred"
echo "  - Compare IoU with frozen (${OUTPUT_DIR}/*/unet_frozen_layer1)"
echo "  - Compare IoU with baseline (0.6377)"
echo ""
