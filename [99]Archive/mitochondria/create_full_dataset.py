#!/usr/bin/env python3
"""
Extract ALL slices from training TIF stacks and create 256x256 patches.
Organizes patches into images/ and masks/ folders for U-Net training.

Based on the existing patch_img.py but modified to:
1. Process ALL slices in the TIF stacks (not just first 12)
2. Create standard dataset structure: images/ and masks/
3. Ensure perfect correspondence between images and masks
"""

import numpy as np
from PIL import Image
import tifffile
from patchify import patchify
import os
from pathlib import Path

def load_tiff_stack_info(file_path):
    """Get information about the TIF stack without loading all data."""
    print(f"Examining TIF stack: {file_path}")
    with tifffile.TiffFile(file_path) as tif:
        # Get basic info
        n_pages = len(tif.pages)
        first_page = tif.pages[0]
        shape = (first_page.shape[0], first_page.shape[1])
        dtype = first_page.dtype

        print(f"  Number of slices: {n_pages}")
        print(f"  Image dimensions: {shape}")
        print(f"  Data type: {dtype}")

        return n_pages, shape, dtype

def extract_all_slices_and_create_patches(training_file, groundtruth_file, output_base_dir, patch_size=256):
    """
    Extract ALL slices from TIF stacks and create patches.

    Args:
        training_file (str): Path to training TIF stack
        groundtruth_file (str): Path to groundtruth TIF stack
        output_base_dir (str): Base directory for output
        patch_size (int): Size of square patches
    """

    # Create output directories
    images_dir = Path(output_base_dir) / "images"
    masks_dir = Path(output_base_dir) / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("FULL TIF STACK PROCESSING - ALL SLICES")
    print("="*80)

    # Get stack information
    train_n_slices, train_shape, train_dtype = load_tiff_stack_info(training_file)
    gt_n_slices, gt_shape, gt_dtype = load_tiff_stack_info(groundtruth_file)

    # Verify stacks match
    if train_n_slices != gt_n_slices:
        raise ValueError(f"Slice count mismatch: training={train_n_slices}, groundtruth={gt_n_slices}")
    if train_shape != gt_shape:
        raise ValueError(f"Image shape mismatch: training={train_shape}, groundtruth={gt_shape}")

    print(f"\n✓ Stacks verified: {train_n_slices} slices, {train_shape} pixels each")

    # Calculate patch grid
    h, w = train_shape
    patches_h = h // patch_size
    patches_w = w // patch_size
    patches_per_slice = patches_h * patches_w
    total_patches = train_n_slices * patches_per_slice

    print(f"\nPatch calculation:")
    print(f"  Original image size: {h} x {w}")
    print(f"  Patches per slice: {patches_h} x {patches_w} = {patches_per_slice}")
    print(f"  Total patches: {train_n_slices} slices × {patches_per_slice} = {total_patches}")

    # Process slices one by one to avoid memory issues
    patch_counter = 0

    print(f"\nProcessing slices...")

    with tifffile.TiffFile(training_file) as train_tif, \
         tifffile.TiffFile(groundtruth_file) as gt_tif:

        for slice_idx in range(train_n_slices):
            print(f"  Slice {slice_idx + 1}/{train_n_slices}", end=" -> ")

            # Load single slice
            train_slice = train_tif.pages[slice_idx].asarray()
            gt_slice = gt_tif.pages[slice_idx].asarray()

            # Crop to make dimensions divisible by patch_size
            cropped_h = patches_h * patch_size
            cropped_w = patches_w * patch_size
            train_cropped = train_slice[:cropped_h, :cropped_w]
            gt_cropped = gt_slice[:cropped_h, :cropped_w]

            # Create patches
            train_patches = patchify(train_cropped, (patch_size, patch_size), step=patch_size)
            gt_patches = patchify(gt_cropped, (patch_size, patch_size), step=patch_size)

            # Reshape from (patches_h, patches_w, patch_size, patch_size) to (n_patches, patch_size, patch_size)
            train_patches = train_patches.reshape(-1, patch_size, patch_size)
            gt_patches = gt_patches.reshape(-1, patch_size, patch_size)

            # Save each patch
            for patch_idx in range(train_patches.shape[0]):
                # Create consistent filename
                filename = f"patch_{patch_counter:06d}.tif"

                # Save training image
                train_patch = train_patches[patch_idx]
                tifffile.imwrite(images_dir / filename, train_patch.astype(np.uint8))

                # Save corresponding mask
                gt_patch = gt_patches[patch_idx]
                tifffile.imwrite(masks_dir / filename, gt_patch.astype(np.uint8))

                patch_counter += 1

            print(f"{train_patches.shape[0]} patches saved")

    print(f"\n✅ PROCESSING COMPLETE!")
    print(f"📊 Total patches created: {patch_counter}")
    print(f"📁 Images saved to: {images_dir}")
    print(f"📁 Masks saved to: {masks_dir}")

    # Create dataset summary
    summary_file = Path(output_base_dir) / "dataset_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Dataset Creation Summary\\n")
        f.write(f"========================\\n\\n")
        f.write(f"Source files:\\n")
        f.write(f"  Training: {training_file}\\n")
        f.write(f"  Groundtruth: {groundtruth_file}\\n\\n")
        f.write(f"Processing parameters:\\n")
        f.write(f"  Patch size: {patch_size}x{patch_size}\\n")
        f.write(f"  Original image size: {h}x{w}\\n")
        f.write(f"  Slices processed: {train_n_slices}\\n\\n")
        f.write(f"Output:\\n")
        f.write(f"  Total patches: {patch_counter}\\n")
        f.write(f"  Images directory: {images_dir}\\n")
        f.write(f"  Masks directory: {masks_dir}\\n\\n")
        f.write(f"Usage:\\n")
        f.write(f"  Update training script to use image_directory = '{images_dir}/'\\n")
        f.write(f"  Update training script to use mask_directory = '{masks_dir}/'\\n")

    print(f"📄 Summary saved to: {summary_file}")

    return patch_counter, images_dir, masks_dir

def verify_dataset_correspondence(images_dir, masks_dir):
    """Verify that each image has a corresponding mask."""
    print(f"\\n🔍 Verifying dataset correspondence...")

    image_files = sorted(list(Path(images_dir).glob("*.tif")))
    mask_files = sorted(list(Path(masks_dir).glob("*.tif")))

    print(f"  Images found: {len(image_files)}")
    print(f"  Masks found: {len(mask_files)}")

    if len(image_files) != len(mask_files):
        print(f"  ❌ MISMATCH: Different number of images and masks!")
        return False

    # Check filename correspondence
    mismatches = 0
    for img_file, mask_file in zip(image_files, mask_files):
        if img_file.name != mask_file.name:
            print(f"  ❌ Name mismatch: {img_file.name} vs {mask_file.name}")
            mismatches += 1

    if mismatches == 0:
        print(f"  ✅ Perfect correspondence: {len(image_files)} image-mask pairs")
        return True
    else:
        print(f"  ❌ Found {mismatches} naming mismatches")
        return False

def main():
    # File paths
    archive_dir = "[99]Archive"
    training_file = f"{archive_dir}/training.tif"
    groundtruth_file = f"{archive_dir}/training_groundtruth.tif"

    # Output directory
    output_base_dir = "dataset_full_stack"

    print(f"🚀 Starting full TIF stack processing...")
    print(f"📁 Input directory: {archive_dir}")
    print(f"📁 Output directory: {output_base_dir}")

    # Check input files exist
    if not os.path.exists(training_file):
        raise FileNotFoundError(f"Training file not found: {training_file}")
    if not os.path.exists(groundtruth_file):
        raise FileNotFoundError(f"Groundtruth file not found: {groundtruth_file}")

    try:
        # Process the full stacks
        patch_count, images_dir, masks_dir = extract_all_slices_and_create_patches(
            training_file, groundtruth_file, output_base_dir, patch_size=256
        )

        # Verify the dataset
        if verify_dataset_correspondence(images_dir, masks_dir):
            print(f"\\n🎉 SUCCESS: Dataset ready for training!")
            print(f"📈 Significant improvement: {patch_count} patches vs previous 144")
            print(f"🔄 Next step: Update training script paths:")
            print(f"   image_directory = '{images_dir}/'")
            print(f"   mask_directory = '{masks_dir}/'")
        else:
            print(f"\\n⚠️  Dataset created but verification failed - check manually")

    except Exception as e:
        print(f"\\n❌ Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()