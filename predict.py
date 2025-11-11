#!/usr/bin/env python3
"""
Standalone script for microbead segmentation using trained U-Net model
======================================================================

This script provides easy-to-use inference for the trained PyTorch U-Net model.
It handles loading, preprocessing, prediction, and post-processing automatically.

Usage:
    # Basic usage
    python predict.py --input image.tif --output mask.png

    # Adjust threshold
    python predict.py --input image.tif --output mask.png --threshold 0.3

    # Force CPU usage
    python predict.py --input image.tif --output mask.png --device cpu

    # Process multiple images
    python predict.py --input-dir ./test_images/ --output-dir ./predictions/

    # Save probability map instead of binary mask
    python predict.py --input image.tif --output prob.png --save-probability

    # Create overlay visualization
    python predict.py --input image.tif --output mask.png --save-overlay

Requirements:
    torch>=2.0.0, numpy>=1.21.0, pillow>=9.0.0

Author: Claude Code
Date: November 11, 2025
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# MODEL ARCHITECTURE (UNet)
# ============================================================================

class ConvBlock(nn.Module):
    """Standard convolution block with BatchNorm and Dropout"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class UNet(nn.Module):
    """Standard UNet architecture for image segmentation"""

    def __init__(self, in_channels: int = 1, n_filters: int = 32, dropout: float = 0.1):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, n_filters, dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(n_filters, n_filters * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(n_filters * 2, n_filters * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(n_filters * 4, n_filters * 8, dropout)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(n_filters * 8, n_filters * 16, dropout)

        # Decoder
        self.up4 = nn.ConvTranspose2d(n_filters * 16, n_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(n_filters * 16, n_filters * 8, dropout)

        self.up3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(n_filters * 8, n_filters * 4, dropout)

        self.up2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(n_filters * 4, n_filters * 2, dropout)

        self.up1 = nn.ConvTranspose2d(n_filters * 2, n_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(n_filters * 2, n_filters, dropout)

        # Output
        self.out = nn.Conv2d(n_filters, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# ============================================================================
# PREPROCESSING
# ============================================================================

def percentile_norm(arr: np.ndarray) -> np.ndarray:
    """
    Percentile normalization (0.5th to 99.5th percentile)
    CRITICAL: Must match training preprocessing!

    Args:
        arr: Input numpy array

    Returns:
        Normalized array in range [0, 1]
    """
    lo, hi = np.percentile(arr, [0.5, 99.5]).astype(np.float32)
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr.astype(np.float32)


def load_and_preprocess_image(
    image_path: Path,
    target_size: int = 512,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Load and preprocess image for model input

    Args:
        image_path: Path to input image
        target_size: Target image size (default 512)
        device: Torch device

    Returns:
        tensor: Preprocessed image tensor [1, 1, H, W]
        original_size: Original image size (width, height)
    """
    # Load as grayscale
    image = Image.open(image_path).convert('L')
    original_size = image.size  # (width, height)

    # Convert to numpy and normalize
    arr = np.array(image, dtype=np.float32)
    arr_norm = percentile_norm(arr)

    # Convert to tensor [1, H, W]
    tensor = torch.from_numpy(arr_norm).unsqueeze(0)

    # Resize to target size if needed
    if tensor.shape[1] != target_size or tensor.shape[2] != target_size:
        tensor = F.interpolate(
            tensor.unsqueeze(0),  # [1, 1, H, W]
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Back to [1, H, W]

    # Add batch dimension and move to device
    tensor = tensor.unsqueeze(0).to(device)  # [1, 1, H, W]

    return tensor, original_size


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(
    model_dir: Path,
    device: torch.device
) -> Tuple[nn.Module, dict]:
    """
    Load trained model from directory

    Args:
        model_dir: Path to model directory
        device: Torch device

    Returns:
        model: Loaded model in eval mode
        model_info: Model metadata dictionary
    """
    # Load model configuration
    model_info_path = model_dir / 'model_info.json'
    if not model_info_path.exists():
        raise FileNotFoundError(f"Model info not found: {model_info_path}")

    with open(model_info_path, 'r') as f:
        model_info = json.load(f)

    # Initialize model with correct hyperparameters
    model = UNet(
        in_channels=1,
        n_filters=model_info['n_filters'],
        dropout=model_info['dropout']
    )

    # Load trained weights
    checkpoint_path = model_dir / 'best_model.pth'
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            f"Please copy the model file from HPC to this location."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, model_info


# ============================================================================
# INFERENCE
# ============================================================================

def predict_image(
    model: nn.Module,
    image_path: Path,
    device: torch.device,
    threshold: float = 0.5,
    target_size: int = 512
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Predict segmentation mask for a single image

    Args:
        model: Trained UNet model
        image_path: Path to input image
        device: Torch device
        threshold: Binarization threshold (0-1)
        target_size: Processing image size

    Returns:
        mask: Binary mask [H, W] with values {0, 1}
        prob_map: Probability map [H, W] with values [0, 1]
        original_size: Original image size (width, height)
    """
    # Load and preprocess
    input_tensor, original_size = load_and_preprocess_image(
        image_path, target_size, device
    )

    # Predict
    with torch.no_grad():
        logits = model(input_tensor)  # [1, 1, H, W]
        prob_map = torch.sigmoid(logits)  # Convert to probabilities

    # Convert to numpy
    prob_map = prob_map.squeeze().cpu().numpy()  # [H, W]

    # Binarize
    mask = (prob_map > threshold).astype(np.uint8)

    return mask, prob_map, original_size


# ============================================================================
# POST-PROCESSING AND VISUALIZATION
# ============================================================================

def create_overlay(
    image_path: Path,
    mask: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create overlay of mask on original image

    Args:
        image_path: Path to original image
        mask: Binary mask [H, W]
        alpha: Overlay transparency (0=invisible, 1=opaque)

    Returns:
        overlay: RGB image with mask overlaid in green
    """
    # Load original
    original = Image.open(image_path).convert('RGB')
    original = np.array(original)

    # Resize mask to match original if needed
    if mask.shape != original.shape[:2]:
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize(
            (original.shape[1], original.shape[0]),
            Image.NEAREST
        )
        mask = np.array(mask_pil)

    # Create green overlay
    overlay = original.copy()
    overlay[mask > 0] = [0, 255, 0]  # Green

    # Blend
    result = (alpha * overlay + (1 - alpha) * original).astype(np.uint8)

    return result


def save_results(
    mask: np.ndarray,
    prob_map: np.ndarray,
    output_path: Path,
    save_probability: bool = False,
    save_overlay: bool = False,
    input_path: Optional[Path] = None
):
    """
    Save prediction results

    Args:
        mask: Binary mask
        prob_map: Probability map
        output_path: Output file path
        save_probability: Save probability map instead of binary mask
        save_overlay: Also save overlay visualization
        input_path: Original input image path (needed for overlay)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_probability:
        # Save probability map (0-255)
        prob_img = (prob_map * 255).astype(np.uint8)
        Image.fromarray(prob_img).save(output_path)
    else:
        # Save binary mask (0 or 255)
        mask_img = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(output_path)

    # Save overlay if requested
    if save_overlay and input_path is not None:
        overlay = create_overlay(input_path, mask)
        overlay_path = output_path.parent / f"{output_path.stem}_overlay.png"
        Image.fromarray(overlay).save(overlay_path)
        print(f"  Saved overlay: {overlay_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Microbead segmentation using trained U-Net model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python predict.py --input image.tif --output mask.png

  # Adjust threshold for more/less sensitive detection
  python predict.py --input image.tif --output mask.png --threshold 0.3

  # Process multiple images in a directory
  python predict.py --input-dir ./test_images/ --output-dir ./predictions/

  # Save probability map (values 0-255 show confidence)
  python predict.py --input image.tif --output prob.png --save-probability

  # Create visualization overlay
  python predict.py --input image.tif --output mask.png --save-overlay

  # Force CPU usage (useful for laptops without GPU)
  python predict.py --input image.tif --output mask.png --device cpu
        """
    )

    # Input/output
    parser.add_argument('--input', type=str,
                       help='Input image path')
    parser.add_argument('--input-dir', type=str,
                       help='Input directory (process all images)')
    parser.add_argument('--output', type=str, default='mask.png',
                       help='Output mask path (default: mask.png)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (for batch processing)')

    # Model
    parser.add_argument('--model-dir', type=str,
                       default='best_models_PyTorch/unet',
                       help='Model directory (default: best_models_PyTorch/unet)')

    # Processing
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Binarization threshold 0-1 (default: 0.5)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cpu, cuda, or auto (default: auto)')
    parser.add_argument('--target-size', type=int, default=512,
                       help='Processing image size (default: 512)')

    # Output options
    parser.add_argument('--save-probability', action='store_true',
                       help='Save probability map instead of binary mask')
    parser.add_argument('--save-overlay', action='store_true',
                       help='Also save overlay visualization')

    args = parser.parse_args()

    # Validate arguments
    if args.input is None and args.input_dir is None:
        parser.error("Either --input or --input-dir must be specified")

    if args.input and args.input_dir:
        parser.error("Cannot specify both --input and --input-dir")

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.model_dir}")
    try:
        model, model_info = load_model(Path(args.model_dir), device)
        print(f"Model loaded successfully!")
        print(f"  Architecture: U-Net")
        print(f"  Filters: {model_info['n_filters']}")
        print(f"  Dropout: {model_info['dropout']}")
        print(f"  Validation IoU: {model_info['best_val_iou']:.2%}")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Single image processing
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        print(f"\nProcessing: {input_path}")

        # Predict
        mask, prob_map, original_size = predict_image(
            model, input_path, device, args.threshold, args.target_size
        )

        # Save
        save_results(
            mask, prob_map, Path(args.output),
            save_probability=args.save_probability,
            save_overlay=args.save_overlay,
            input_path=input_path
        )

        # Stats
        foreground_pixels = mask.sum()
        total_pixels = mask.size
        foreground_pct = 100 * foreground_pixels / total_pixels

        print(f"  Saved: {args.output}")
        print(f"  Foreground: {foreground_pixels:,} pixels ({foreground_pct:.2f}%)")

    # Batch directory processing
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
            sys.exit(1)

        output_dir = Path(args.output_dir) if args.output_dir else Path('predictions')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'}
        image_files = [
            f for f in sorted(input_dir.iterdir())
            if f.suffix.lower() in image_exts
        ]

        if not image_files:
            print(f"No images found in {input_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"\nProcessing {len(image_files)} images from: {input_dir}")
        print(f"Output directory: {output_dir}")

        # Process each image
        for i, input_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {input_path.name}")

            try:
                # Predict
                mask, prob_map, original_size = predict_image(
                    model, input_path, device, args.threshold, args.target_size
                )

                # Determine output filename
                if args.save_probability:
                    output_name = f"{input_path.stem}_prob.png"
                else:
                    output_name = f"{input_path.stem}_mask.png"

                output_path = output_dir / output_name

                # Save
                save_results(
                    mask, prob_map, output_path,
                    save_probability=args.save_probability,
                    save_overlay=args.save_overlay,
                    input_path=input_path
                )

                # Stats
                foreground_pixels = mask.sum()
                total_pixels = mask.size
                foreground_pct = 100 * foreground_pixels / total_pixels

                print(f"  Saved: {output_path}")
                print(f"  Foreground: {foreground_pixels:,} pixels ({foreground_pct:.2f}%)")

            except Exception as e:
                print(f"  Error processing {input_path.name}: {e}", file=sys.stderr)
                continue

        print(f"\nCompleted! Results saved to: {output_dir}")

    print("\nDone!")


if __name__ == '__main__':
    main()
