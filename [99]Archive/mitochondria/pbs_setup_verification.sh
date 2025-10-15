#!/bin/bash
# Verification script for mitochondria segmentation setup

echo "=========================================="
echo "MITOCHONDRIA SEGMENTATION SETUP VERIFICATION"
echo "=========================================="
echo ""

# Check dataset structure
echo "1. DATASET STRUCTURE CHECK:"
echo "=========================="

if [ -d "./dataset" ]; then
    echo "✓ ./dataset/ directory exists"

    if [ -d "./dataset/images" ]; then
        echo "✓ ./dataset/images/ directory exists"
        img_count=$(find ./dataset/images/ -name "*.tif" | wc -l)
        echo "  - Found $img_count .tif image files"

        if [ $img_count -gt 0 ]; then
            echo "  - Sample image files:"
            find ./dataset/images/ -name "*.tif" | head -3
        else
            echo "  ⚠️  WARNING: No .tif files found in images directory"
        fi
    else
        echo "❌ ./dataset/images/ directory missing"
    fi

    if [ -d "./dataset/masks" ]; then
        echo "✓ ./dataset/masks/ directory exists"
        mask_count=$(find ./dataset/masks/ -name "*.tif" | wc -l)
        echo "  - Found $mask_count .tif mask files"

        if [ $mask_count -gt 0 ]; then
            echo "  - Sample mask files:"
            find ./dataset/masks/ -name "*.tif" | head -3
        else
            echo "  ⚠️  WARNING: No .tif files found in masks directory"
        fi

        # Check if image and mask counts match
        if [ $img_count -eq $mask_count ] && [ $img_count -gt 0 ]; then
            echo "✓ Image and mask file counts match ($img_count each)"
        elif [ $img_count -ne $mask_count ]; then
            echo "⚠️  WARNING: Mismatch between images ($img_count) and masks ($mask_count)"
        fi
    else
        echo "❌ ./dataset/masks/ directory missing"
    fi
else
    echo "❌ ./dataset/ directory not found"
    echo ""
    echo "EXPECTED DATASET STRUCTURE:"
    echo "  ./dataset/"
    echo "  ├── images/"
    echo "  │   ├── image1.tif"
    echo "  │   ├── image2.tif"
    echo "  │   └── ..."
    echo "  └── masks/"
    echo "      ├── mask1.tif"
    echo "      ├── mask2.tif"
    echo "      └── ..."
fi

echo ""

# Check Python files
echo "2. PYTHON FILES CHECK:"
echo "====================="

required_files=("224_225_226_mito_segm_using_various_unet_models.py" "224_225_226_models.py")

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file exists"
        size=$(stat -c%s "$file")
        echo "  - Size: $size bytes"
    else
        echo "❌ $file missing"
    fi
done

echo ""

# Check TensorFlow container
echo "3. TENSORFLOW CONTAINER CHECK:"
echo "============================="

TF_CONTAINER="/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif"
if [ -f "$TF_CONTAINER" ]; then
    echo "✓ TensorFlow container found"
    echo "  - Path: $TF_CONTAINER"
    size_gb=$(du -h "$TF_CONTAINER" | cut -f1)
    echo "  - Size: $size_gb"
else
    echo "❌ TensorFlow container not found at expected location"
    echo "Available TensorFlow containers:"
    ls -la /app1/common/singularity-img/hopper/tensorflow/ 2>/dev/null || echo "  Directory not accessible"
fi

echo ""

# Test container basic functionality
echo "4. CONTAINER FUNCTIONALITY TEST:"
echo "==============================="

if [ -f "$TF_CONTAINER" ]; then
    echo "Testing basic Python and TensorFlow import..."

    module load singularity 2>/dev/null

    singularity exec --nv "$TF_CONTAINER" python3 -c "
import sys
print('✓ Python version:', sys.version.split()[0])

try:
    import tensorflow as tf
    print('✓ TensorFlow version:', tf.__version__)
    print('✓ CUDA support:', tf.test.is_built_with_cuda())

    gpus = tf.config.list_physical_devices('GPU')
    print('✓ GPUs detected:', len(gpus))

except Exception as e:
    print('❌ TensorFlow test failed:', e)

# Test other key dependencies
deps = ['cv2', 'numpy', 'matplotlib', 'PIL', 'sklearn', 'pandas']
print()
print('Dependencies check:')
for dep in deps:
    try:
        __import__(dep)
        print(f'✓ {dep}')
    except ImportError:
        print(f'❌ {dep}')
" 2>/dev/null || echo "❌ Container test failed - check module loading and container path"
else
    echo "❌ Cannot test - container not found"
fi

echo ""

# Disk space check
echo "5. DISK SPACE CHECK:"
echo "=================="

echo "Current directory disk usage:"
du -sh . 2>/dev/null || echo "Unable to check disk usage"

echo "Available space in current filesystem:"
df -h . | tail -1 | awk '{print "  Available: " $4 " (" $5 " used)"}'

echo ""

# Final recommendations
echo "6. SETUP RECOMMENDATIONS:"
echo "======================="

if [ -d "./dataset/images" ] && [ -d "./dataset/masks" ] && [ -f "$TF_CONTAINER" ]; then
    echo "✓ Basic setup appears correct"
    echo ""
    echo "READY TO SUBMIT JOB:"
    echo "  qsub mitochondria_segmentation.sh"
    echo ""
    echo "ESTIMATED TRAINING TIME: 8-12 hours"
    echo "ESTIMATED DISK USAGE: ~2-5 GB for model files and logs"
else
    echo "❌ Setup issues detected - fix the above problems before submitting job"
    echo ""
    echo "REQUIRED FIXES:"

    if [ ! -d "./dataset" ]; then
        echo "  1. Create ./dataset/ directory with images/ and masks/ subdirectories"
        echo "  2. Add your .tif image and mask files to respective directories"
    fi

    if [ ! -f "224_225_226_mito_segm_using_various_unet_models.py" ]; then
        echo "  3. Ensure Python training script is in current directory"
    fi

    if [ ! -f "$TF_CONTAINER" ]; then
        echo "  4. Verify TensorFlow container path or contact HPC support"
    fi
fi

echo ""
echo "=========================================="
echo "VERIFICATION COMPLETE"
echo "=========================================="
