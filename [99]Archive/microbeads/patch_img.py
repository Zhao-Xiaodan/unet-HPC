import numpy as np
from PIL import Image
import tifffile
from patchify import patchify
import os

def load_and_process_tiff_stack(file_path, num_slices=12, patch_size=256):
    """
    Load a TIFF stack, extract first num_slices, and create 256x256 patches

    Args:
        file_path (str): Path to the TIFF file
        num_slices (int): Number of slices to extract from the beginning
        patch_size (int): Size of square patches to create

    Returns:
        numpy.ndarray: Array of patches with shape (n_patches, patch_size, patch_size)
    """
    # Load the TIFF stack
    print(f"Loading TIFF stack from: {file_path}")
    with tifffile.TiffFile(file_path) as tif:
        # Read all pages/slices
        images = tif.asarray()
        print(f"Original stack shape: {images.shape}")

        # Take only the first num_slices
        if len(images.shape) == 3:  # (slices, height, width)
            selected_slices = images[:num_slices]
        else:
            raise ValueError(f"Expected 3D array (slices, height, width), got shape: {images.shape}")

        print(f"Selected {num_slices} slices, new shape: {selected_slices.shape}")

    # Process each slice and create patches
    all_patches = []

    for i, slice_img in enumerate(selected_slices):
        print(f"Processing slice {i+1}/{num_slices}")

        # Ensure the image dimensions are divisible by patch_size
        h, w = slice_img.shape

        # Calculate how many complete patches we can get
        patches_h = h // patch_size
        patches_w = w // patch_size

        # Crop to make dimensions divisible by patch_size
        cropped_h = patches_h * patch_size
        cropped_w = patches_w * patch_size

        cropped_slice = slice_img[:cropped_h, :cropped_w]
        print(f"  Original slice shape: {slice_img.shape}")
        print(f"  Cropped slice shape: {cropped_slice.shape}")
        print(f"  Will create {patches_h}x{patches_w} = {patches_h*patches_w} patches")

        # Create patches using patchify
        patches = patchify(cropped_slice, (patch_size, patch_size), step=patch_size)

        # Reshape patches from (patches_h, patches_w, patch_size, patch_size)
        # to (n_patches, patch_size, patch_size)
        patches = patches.reshape(-1, patch_size, patch_size)

        all_patches.append(patches)
        print(f"  Created {patches.shape[0]} patches")

    # Combine all patches
    all_patches = np.concatenate(all_patches, axis=0)
    print(f"\nTotal patches created: {all_patches.shape[0]}")
    print(f"Final patches shape: {all_patches.shape}")

    return all_patches

def save_patches(patches, output_dir, prefix):
    """
    Save patches as individual TIFF files

    Args:
        patches (numpy.ndarray): Array of patches
        output_dir (str): Directory to save patches
        prefix (str): Prefix for filenames
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, patch in enumerate(patches):
        filename = f"{prefix}_patch_{i:04d}.tif"
        filepath = os.path.join(output_dir, filename)
        tifffile.imwrite(filepath, patch.astype(np.uint8))

    print(f"Saved {len(patches)} patches to {output_dir}")

def main():
    # File paths
    training_file = "[99]Archive/training.tif"
    groundtruth_file = "[99]Archive/training_groundtruth.tif"

    # Parameters
    num_slices = 12
    patch_size = 256

    # Create output directories
    output_dir = "processed_patches"
    training_output_dir = os.path.join(output_dir, "training")
    groundtruth_output_dir = os.path.join(output_dir, "groundtruth")

    # Process training data
    print("=" * 50)
    print("PROCESSING TRAINING DATA")
    print("=" * 50)
    training_patches = load_and_process_tiff_stack(
        training_file, num_slices=num_slices, patch_size=patch_size
    )

    # Process ground truth data
    print("\n" + "=" * 50)
    print("PROCESSING GROUND TRUTH DATA")
    print("=" * 50)
    groundtruth_patches = load_and_process_tiff_stack(
        groundtruth_file, num_slices=num_slices, patch_size=patch_size
    )

    # Verify both have the same number of patches
    if training_patches.shape[0] != groundtruth_patches.shape[0]:
        print(f"WARNING: Number of patches don't match!")
        print(f"Training patches: {training_patches.shape[0]}")
        print(f"Ground truth patches: {groundtruth_patches.shape[0]}")
    else:
        print(f"\nSUCCESS: Both files produced {training_patches.shape[0]} patches")

    # Save patches
    print("\n" + "=" * 50)
    print("SAVING PATCHES")
    print("=" * 50)
    save_patches(training_patches, training_output_dir, "training")
    save_patches(groundtruth_patches, groundtruth_output_dir, "groundtruth")

    # Save as numpy arrays for easy loading
    np.save(os.path.join(output_dir, "training_patches.npy"), training_patches)
    np.save(os.path.join(output_dir, "groundtruth_patches.npy"), groundtruth_patches)

    print(f"\nAlso saved as numpy arrays:")
    print(f"  {os.path.join(output_dir, 'training_patches.npy')}")
    print(f"  {os.path.join(output_dir, 'groundtruth_patches.npy')}")

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
