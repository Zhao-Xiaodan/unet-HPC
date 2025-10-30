#!/usr/bin/bash
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -N EdgeDet_Viz
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# Edge Detector Experiment: Batch Feature Visualization
# ==============================================================================
#
# Generate feature visualizations for all three models:
# 1. Baseline (random initialization) - skip if already exists
# 2. Frozen layer 1 (Gabor filters frozen)
# 3. Trainable layer 1 (Gabor filters trainable)
#
# This enables direct comparison of layer 1 features across strategies.
#
# Expected runtime: ~6-8 hours (3 models × 6 layers × ~20 min/layer)
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

echo "Edge Detector Experiment: Batch Visualization"
echo ""
echo "Models to visualize:"
echo "  1. Baseline (random init) - if not already done"
echo "  2. Frozen layer 1 (Gabor init, frozen)"
echo "  3. Trainable layer 1 (Gabor init, trainable)"
echo ""
echo "Key comparison: encoder_1_conv2"
echo "  - Baseline: Textures (current observation)"
echo "  - Frozen: Pure Gabor filters (edges)"
echo "  - Trainable: Gabor → ? (adaptation)"
echo ""

echo "========================================"
echo "Starting batch visualization..."
echo "========================================"
echo ""

# Run batch visualization (skip baseline if already exists)
singularity exec --nv $image python visualize_edge_detector_experiment.py \
    --skip_baseline \
    --baseline_viz_dir ./unet_feature_viz_20251029_065244 \
    --n_filters 32 \
    --dropout 0.2 \
    --channels_per_layer 12 \
    --diverse_per_channel 3 \
    --iterations 500

echo ""
echo "========================================"
echo "Batch visualization complete!"
echo "End Time: $(date)"
echo "========================================"
echo ""

echo "Results locations:"
echo "  Baseline: ./unet_feature_viz_20251029_065244/"
echo "  Frozen: ./edge_detector_experiment/*/visualizations/frozen_layer1/"
echo "  Trainable: ./edge_detector_experiment/*/visualizations/trainable_layer1/"
echo ""

echo "Next steps:"
echo "1. Download visualizations to local machine"
echo "2. Compare encoder_1_conv2 across all three models"
echo "3. Run analysis:"
echo "   python analyze_edge_detector_experiment.py"
echo ""
