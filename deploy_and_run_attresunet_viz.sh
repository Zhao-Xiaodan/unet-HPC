#!/bin/bash
# ==============================================================================
# Quick Deployment Script: Attention ResU-Net Feature Visualization
# ==============================================================================
#
# This script pulls the latest fixes and submits the Attention ResU-Net
# visualization job to HPC.
#
# FIXED: n_filters mismatch (32 → 64) - Job 330076 should now succeed
#
# Usage on HPC:
#   cd ~/scratch/unet-HPC
#   bash deploy_and_run_attresunet_viz.sh
# ==============================================================================

set -e  # Exit on error

echo "========================================"
echo "Attention ResU-Net Visualization Deploy"
echo "========================================"
echo ""

# Pull latest changes
echo "📥 Pulling latest changes from GitHub..."
git pull

echo ""
echo "✅ Latest fixes applied:"
echo "   - attention_resunet_feature_visualization.py: n_filters=64"
echo "   - pbs_attention_resunet_feature_viz.sh: N_FILTERS=64"
echo ""

# Verify fix is in place
N_FILTERS_VALUE=$(grep "^N_FILTERS=" pbs_attention_resunet_feature_viz.sh | cut -d'=' -f2 | cut -d'#' -f1 | tr -d ' ')

if [ "$N_FILTERS_VALUE" != "64" ]; then
    echo "❌ ERROR: N_FILTERS is not 64 (found: $N_FILTERS_VALUE)"
    echo "   Please check pbs_attention_resunet_feature_viz.sh"
    exit 1
fi

echo "✓ Verification passed: N_FILTERS=$N_FILTERS_VALUE"
echo ""

# Submit job
echo "🚀 Submitting Attention ResU-Net visualization job..."
JOB_ID=$(qsub pbs_attention_resunet_feature_viz.sh)

echo ""
echo "========================================"
echo "✅ JOB SUBMITTED: $JOB_ID"
echo "========================================"
echo ""
echo "Monitor with:"
echo "  qstat -u \$USER"
echo ""
echo "Watch log:"
echo "  tail -f AttResUNet_Feat_Viz.o*"
echo ""
echo "Expected output (if fix worked):"
echo "  ✓ Model loaded (epoch XX)"
echo "  ✓ Best validation IoU: 0.XXXX"
echo "  Layer: encoder_1_resconv"
echo "  Optimizing: 100%|██████| 500/500"
echo ""
echo "Previous error (Job 330076) was:"
echo "  RuntimeError: size mismatch [64, 1, 3, 3] vs [32, 1, 3, 3]"
echo ""
echo "This has been FIXED with n_filters=64"
echo "========================================"
