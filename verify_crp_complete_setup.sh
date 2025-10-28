#!/bin/bash
# ==============================================================================
# CRP Complete Setup Verification Script
# ==============================================================================
# This script verifies all required files and configurations are in place
# before running the complete CRP analysis.
# ==============================================================================

echo "========================================"
echo "CRP Complete Setup Verification"
echo "========================================"
echo ""

ERRORS=0
WARNINGS=0

# Check Python scripts
echo "[1/6] Checking analysis scripts..."
if [ -f "unet_crp_complete_with_encoders.py" ]; then
    echo "  ✓ unet_crp_complete_with_encoders.py found"
else
    echo "  ✗ unet_crp_complete_with_encoders.py NOT FOUND"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "generate_enhanced_interactive_visualization.py" ]; then
    echo "  ✓ generate_enhanced_interactive_visualization.py found"
else
    echo "  ✗ generate_enhanced_interactive_visualization.py NOT FOUND"
    ERRORS=$((ERRORS + 1))
fi

# Check PBS script
echo ""
echo "[2/6] Checking PBS submission script..."
if [ -f "pbs_crp_complete_with_encoders.sh" ]; then
    echo "  ✓ pbs_crp_complete_with_encoders.sh found"
    if [ -x "pbs_crp_complete_with_encoders.sh" ]; then
        echo "  ✓ Script is executable"
    else
        echo "  ⚠ Script is not executable (fixing...)"
        chmod +x pbs_crp_complete_with_encoders.sh
        echo "  ✓ Made executable"
    fi
else
    echo "  ✗ pbs_crp_complete_with_encoders.sh NOT FOUND"
    ERRORS=$((ERRORS + 1))
fi

# Check model
echo ""
echo "[3/6] Checking trained model..."
if [ -f "./best_models_PyTorch/unet/best_model.pth" ]; then
    MODEL_SIZE=$(du -h "./best_models_PyTorch/unet/best_model.pth" | cut -f1)
    echo "  ✓ Model found (size: $MODEL_SIZE)"
else
    echo "  ✗ Model NOT FOUND at ./best_models_PyTorch/unet/best_model.pth"
    ERRORS=$((ERRORS + 1))
fi

# Check test images
echo ""
echo "[4/6] Checking test images..."
if [ -d "./test_images" ]; then
    NUM_TIF=$(ls ./test_images/*.tif 2>/dev/null | wc -l)
    NUM_TIFF=$(ls ./test_images/*.tiff 2>/dev/null | wc -l)
    TOTAL=$((NUM_TIF + NUM_TIFF))

    if [ $TOTAL -gt 0 ]; then
        echo "  ✓ Test images directory found"
        echo "  ✓ Found $TOTAL test images (.tif/.tiff)"

        if [ $TOTAL -gt 8 ]; then
            echo "  ⚠ Warning: $TOTAL images will take ~$((TOTAL * 50 / 60)) hours to process"
            WARNINGS=$((WARNINGS + 1))
        else
            ESTIMATED_TIME=$((TOTAL * 50 / 60))
            echo "  ℹ Estimated runtime: ~$ESTIMATED_TIME hours"
        fi
    else
        echo "  ✗ No test images found in ./test_images/"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  ✗ Test images directory NOT FOUND"
    ERRORS=$((ERRORS + 1))
fi

# Check Python dependencies
echo ""
echo "[5/6] Checking Python dependencies..."
if command -v python &> /dev/null; then
    echo "  ✓ Python found"

    # Check if we can import key modules (won't work without singularity, but good to check)
    python -c "import torch; import numpy; import matplotlib; import PIL; import tqdm" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✓ All Python dependencies available"
    else
        echo "  ⚠ Some dependencies not available locally (will use singularity container)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ⚠ Python not found locally (will use singularity container)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check disk space
echo ""
echo "[6/6] Checking disk space..."
AVAILABLE_GB=$(df -H . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ ! -z "$AVAILABLE_GB" ]; then
    echo "  ℹ Available disk space: ${AVAILABLE_GB}GB"

    # Estimate required space: ~1GB per image for full analysis
    REQUIRED_GB=$((TOTAL + 5))

    if [ $(echo "$AVAILABLE_GB > $REQUIRED_GB" | bc -l) -eq 1 ]; then
        echo "  ✓ Sufficient disk space (need ~${REQUIRED_GB}GB)"
    else
        echo "  ⚠ Low disk space (need ~${REQUIRED_GB}GB, have ${AVAILABLE_GB}GB)"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

# Summary
echo ""
echo "========================================"
echo "Verification Summary"
echo "========================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✅ All checks passed! Ready to submit job."
    echo ""
    echo "To submit the job, run:"
    echo "  qsub pbs_crp_complete_with_encoders.sh"
    echo ""
    EXIT_CODE=0
elif [ $ERRORS -eq 0 ]; then
    echo "⚠️  Setup OK with $WARNINGS warning(s)"
    echo ""
    echo "Warnings are not critical. You can proceed:"
    echo "  qsub pbs_crp_complete_with_encoders.sh"
    echo ""
    EXIT_CODE=0
else
    echo "❌ Found $ERRORS error(s) and $WARNINGS warning(s)"
    echo ""
    echo "Please fix errors before submitting job."
    echo ""
    EXIT_CODE=1
fi

echo "For detailed usage instructions, see:"
echo "  CRP_COMPLETE_USAGE_GUIDE.md"
echo "========================================"

exit $EXIT_CODE
