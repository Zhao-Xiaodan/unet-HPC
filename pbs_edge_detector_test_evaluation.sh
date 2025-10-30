#!/usr/bin/bash
#PBS -N EdgeDet_TestEval
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -j oe
#PBS -k oed
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

################################################################################
# Edge Detector Experiment: Test Set Evaluation
#
# Evaluates three models on 8 held-out test images (./test_images):
#   1. Baseline (random initialization)
#   2. Frozen Layer 1 (Gabor edges, frozen)
#   3. Trainable Layer 1 (Gabor edges, trainable)
#
# Outputs:
#   - Prediction visualizations for each image (8 images × 3 models = 24 files)
#   - Comparison plot (bar chart)
#   - JSON results with predicted cell counts
#
# Expected runtime: ~10 minutes (inference on 24 images)
################################################################################

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
BASELINE_MODEL="./best_models_PyTorch/unet/best_model.pth"
FROZEN_MODEL="./edge_detector_experiment/20251030_122713/unet_frozen_layer1/best_model.pth"
TRAINABLE_MODEL="./edge_detector_experiment/20251030_122737/unet_trainable_layer1/best_model.pth"
TEST_IMAGES_DIR="./test_images"
OUTPUT_DIR="./edge_detector_test_evaluation"
N_FILTERS=32
DROPOUT=0.2
TILE_ROW=3
TILE_COL=4

echo "Configuration:"
echo "  Baseline model: $BASELINE_MODEL"
echo "  Frozen model: $FROZEN_MODEL"
echo "  Trainable model: $TRAINABLE_MODEL"
echo "  Test images: $TEST_IMAGES_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""

echo "Test set details:"
echo "  8 images at different dilutions (80x, 160x, 320x, 640x, 1280x, 2560x, 5120x, 10240x)"
echo "  These images were NOT used during training (held-out test set)"
echo "  Provides unbiased estimate of generalization performance"
echo ""

echo "Expected outputs:"
echo "  ✓ 8 prediction images × 3 models = 24 visualization files"
echo "  ✓ Comparison plot (bar chart across dilutions)"
echo "  ✓ JSON results with predicted cell counts"
echo ""

echo "Key questions to answer:"
echo "  1. Do Gabor-initialized models generalize better to unseen data?"
echo "  2. How do models perform at extreme dilutions (80x, 10240x)?"
echo "  3. Is frozen layer 1 sufficient or does trainable perform better?"
echo ""

echo "========================================"
echo "Starting test set evaluation..."
echo "========================================"
echo ""

# Run evaluation
singularity exec --nv $image python evaluate_edge_detector_on_test.py \
    --baseline_model $BASELINE_MODEL \
    --frozen_model $FROZEN_MODEL \
    --trainable_model $TRAINABLE_MODEL \
    --test_images_dir $TEST_IMAGES_DIR \
    --output_dir $OUTPUT_DIR \
    --n_filters $N_FILTERS \
    --dropout $DROPOUT \
    --tile_row $TILE_ROW \
    --tile_col $TILE_COL

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "End Time: $(date)"
echo "========================================"
echo ""

# Show output directory structure
echo "Output directory structure:"
ls -la $OUTPUT_DIR/
echo ""

# Show results for each model
for model in baseline frozen_layer1 trainable_layer1; do
    if [ -d "$OUTPUT_DIR/$model" ]; then
        echo "Model: $model"
        echo "  Images generated:"
        ls -1 $OUTPUT_DIR/$model/*.png 2>/dev/null | wc -l | xargs echo "    " | tr '\n' ' ' && echo "prediction files"
        echo ""
    fi
done

# Show comparison results
if [ -f "$OUTPUT_DIR/comparison_summary.json" ]; then
    echo "Comparison summary:"
    cat $OUTPUT_DIR/comparison_summary.json | head -n 30
    echo ""
fi

if [ -f "$OUTPUT_DIR/comparison_plot.png" ]; then
    echo "✓ Comparison plot generated: $OUTPUT_DIR/comparison_plot.png"
fi

echo "========================================"
echo "✓ Job complete!"
echo "========================================"
echo ""

echo "Next steps:"
echo "1. Download results:"
echo "   scp -r <username>@hopper:~/scratch/unet-HPC/$OUTPUT_DIR ./"
echo ""
echo "2. View comparison plot:"
echo "   open $OUTPUT_DIR/comparison_plot.png"
echo ""
echo "3. Analyze JSON results:"
echo "   cat $OUTPUT_DIR/comparison_summary.json"
echo ""
echo "4. Compare individual predictions:"
echo "   open $OUTPUT_DIR/*/320x_2025-05-15_02-05-00_prediction.png"
echo ""
echo "Expected findings:"
echo "  - If Gabor models (frozen/trainable) predict better: Edge detectors help generalization"
echo "  - If baseline predicts better: Texture features may be more robust"
echo "  - If frozen ≈ trainable: Layer 1 adaptation had minimal effect on generalization"
