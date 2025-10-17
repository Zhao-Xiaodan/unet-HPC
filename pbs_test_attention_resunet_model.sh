#!/bin/bash
#PBS -l walltime=0:10:00
#PBS -j oe
#PBS -k oed
#PBS -N Test_AttnResUNet_Model
#PBS -l select=1:ncpus=4:mpiprocs=1:ompthreads=4:ngpus=1:mem=32gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

################################################################################
# Attention ResUNet Model Loading Diagnostic Test
################################################################################
#
# Purpose: Diagnose why density analysis crashes at 0% prediction
#
# Tests:
#   1. Model loading with custom objects
#   2. Single tile prediction
#   3. Batch prediction (4 tiles)
#   4. Full image prediction (28 tiles)
#   5. Model architecture summary
#
# Expected runtime: < 5 minutes
#
# Date: October 17, 2025
################################################################################

echo "========================================================================"
echo "ATTENTION RESUNET MODEL LOADING DIAGNOSTIC TEST"
echo "========================================================================"
echo "Job ID: $PBS_JOBID"
echo "Job Name: $PBS_JOBNAME"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $PBS_O_WORKDIR"
echo "========================================================================"
echo ""

# Navigate to working directory
cd /home/svu/phyzxi/scratch/unet-HPC

# =======================================================================
# ENVIRONMENT SETUP
# =======================================================================

echo "=== ENVIRONMENT SETUP ==="

export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export CUDA_VISIBLE_DEVICES=0

module load singularity

image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

if [ ! -f "$image" ]; then
    echo "ERROR: TensorFlow container not found at $image"
    exit 1
fi

echo "✓ TensorFlow Container: $image"
echo "============================"
echo ""

# Verify test script exists
TEST_SCRIPT="./test_attention_resunet_model_loading.py"

if [ ! -f "$TEST_SCRIPT" ]; then
    echo "ERROR: Test script not found: $TEST_SCRIPT"
    exit 1
fi

echo "✓ Test script: $TEST_SCRIPT"
echo ""

# Run diagnostic test
echo "========================================================================"
echo "RUNNING DIAGNOSTIC TEST"
echo "========================================================================"
echo "Command: singularity exec --nv \"$image\" python3 $TEST_SCRIPT"
echo ""

singularity exec --nv "$image" python3 "$TEST_SCRIPT"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "DIAGNOSTIC TEST COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ All tests passed!"
    echo ""
    echo "Conclusion: Model loads and predicts correctly in isolation."
    echo "The issue is likely in the density analysis script itself,"
    echo "not in the model or custom objects."
else
    echo ""
    echo "✗ Tests failed with exit code $EXIT_CODE"
    echo "Check the output above for details."
fi

echo "========================================================================"

exit $EXIT_CODE
