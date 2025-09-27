#!/usr/bin/env python3
"""
Setup script to ensure the full dataset is available for training.
Creates dataset_full_stack if needed, and sets up compatibility symlinks.
"""

import os
import sys
from pathlib import Path

def check_dataset_status():
    """Check what dataset options are available."""
    print("🔍 Checking dataset availability...")

    # Check for full stack dataset
    if Path("dataset_full_stack/images").exists() and Path("dataset_full_stack/masks").exists():
        image_count = len(list(Path("dataset_full_stack/images").glob("*.tif")))
        mask_count = len(list(Path("dataset_full_stack/masks").glob("*.tif")))
        print(f"✅ Full stack dataset found: {image_count} images, {mask_count} masks")
        return "full_stack", image_count

    # Check for source TIF files
    elif Path("[99]Archive/training.tif").exists() and Path("[99]Archive/training_groundtruth.tif").exists():
        print("⚠️  Full stack dataset not found, but source TIF files available")
        return "tif_files", 0

    # Check for old dataset
    elif Path("dataset/images").exists() and Path("dataset/masks").exists():
        image_count = len(list(Path("dataset/images").glob("*.tif")))
        mask_count = len(list(Path("dataset/masks").glob("*.tif")))
        print(f"⚠️  Using old dataset: {image_count} images, {mask_count} masks")
        return "old_dataset", image_count

    else:
        print("❌ No dataset found!")
        return "none", 0

def create_full_dataset():
    """Create the full dataset from TIF stacks."""
    print("🚀 Creating full dataset from TIF stacks...")

    try:
        # Import and run the dataset creation
        import subprocess
        result = subprocess.run([sys.executable, "create_full_dataset.py"],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Full dataset created successfully!")
            return True
        else:
            print(f"❌ Dataset creation failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return False

def setup_compatibility_links():
    """Set up compatibility symlinks for training scripts."""
    print("🔗 Setting up compatibility links...")

    # Remove old symlinks if they exist
    if Path("data").is_symlink():
        Path("data").unlink()
        print("  Removed old 'data' symlink")

    # Create new symlink pointing to full dataset
    if Path("dataset_full_stack").exists():
        os.symlink("dataset_full_stack", "data")
        print("  Created 'data' -> 'dataset_full_stack' symlink")
        return True
    else:
        print("  ❌ Cannot create symlink - dataset_full_stack not found")
        return False

def main():
    print("=" * 60)
    print("DATASET SETUP FOR MITOCHONDRIA SEGMENTATION")
    print("=" * 60)

    # Check current status
    status, count = check_dataset_status()

    if status == "full_stack":
        print(f"✅ Full dataset ready with {count} patches!")

    elif status == "tif_files":
        print("Creating full dataset from TIF stacks...")
        if create_full_dataset():
            status, count = check_dataset_status()
            print(f"✅ Full dataset created with {count} patches!")
        else:
            print("❌ Failed to create full dataset")
            sys.exit(1)

    elif status == "old_dataset":
        print(f"⚠️  Using old dataset with only {count} patches")
        print("  Consider running create_full_dataset.py for better results")

    else:
        print("❌ No dataset available and cannot create one")
        print("  Required files:")
        print("  - [99]Archive/training.tif")
        print("  - [99]Archive/training_groundtruth.tif")
        print("  - create_full_dataset.py")
        sys.exit(1)

    # Set up compatibility links
    if not setup_compatibility_links():
        print("⚠️  Symlink setup failed - training may need manual path adjustment")

    print("\n🎯 DATASET SETUP COMPLETE!")
    print(f"📊 Training will use {count} image-mask pairs")
    print("🚀 Ready for training job submission!")

if __name__ == "__main__":
    main()