#!/usr/bin/env python3
"""
Evaluate Edge Detector Experiment Models on Test Images

This script evaluates the frozen and trainable Gabor-initialized models
on the 8 test images (./test_images/) that were NOT used during training.

Comparison:
- dataset_shrunk_masks: 98 images (78 train, 20 val) - used for training
- test_images: 8 images (320x, 640x, 1280x, etc.) - held out for testing

This answers: "How well do Gabor-initialized models generalize to unseen data?"
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================================
# MODEL ARCHITECTURE (Must match training)
# ============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and Dropout"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            x = self.dropout(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    """Standard U-Net architecture (matches training)"""
    def __init__(self, in_channels=1, n_filters=32, dropout=0.2):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, n_filters, dropout=dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(n_filters, n_filters*2, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(n_filters*2, n_filters*4, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(n_filters*4, n_filters*8, dropout=dropout)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(n_filters*8, n_filters*16, dropout=dropout)

        # Decoder
        self.up4 = nn.ConvTranspose2d(n_filters*16, n_filters*8, 2, stride=2)
        self.dec4 = ConvBlock(n_filters*16, n_filters*8, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(n_filters*8, n_filters*4, 2, stride=2)
        self.dec3 = ConvBlock(n_filters*8, n_filters*4, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(n_filters*4, n_filters*2, 2, stride=2)
        self.dec2 = ConvBlock(n_filters*4, n_filters*2, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(n_filters*2, n_filters, 2, stride=2)
        self.dec1 = ConvBlock(n_filters*2, n_filters, dropout=dropout)

        # Output
        self.out = nn.Conv2d(n_filters, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def extract_center_tile(image_array, tile_size=512, row=3, col=4):
    """Extract center 512x512 tile from 5×8 grid"""
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = np.mean(image_array, axis=2)  # Convert RGB to grayscale

    img_height, img_width = image_array.shape
    tile_height = img_height // 5
    tile_width = img_width // 8

    top = row * tile_height
    left = col * tile_width

    # Extract tile and resize to 512×512
    tile = image_array[top:top+tile_height, left:left+tile_width]
    from PIL import Image as PILImage
    tile_pil = PILImage.fromarray((tile * 255).astype(np.uint8) if tile.max() <= 1 else tile.astype(np.uint8))
    tile_resized = tile_pil.resize((tile_size, tile_size), PILImage.LANCZOS)
    return np.array(tile_resized).astype(np.float32) / 255.0

def preprocess_tile(tile_array):
    """Convert tile to PyTorch tensor with normalization"""
    # Normalize to [0, 1]
    tile_normalized = tile_array.astype(np.float32)
    if tile_normalized.max() > 1:
        tile_normalized /= 255.0

    # Add batch and channel dimensions: [1, 1, 512, 512]
    tile_tensor = torch.from_numpy(tile_normalized).unsqueeze(0).unsqueeze(0)
    return tile_tensor

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_iou(pred, target, threshold=0.5):
    """Compute IoU (Jaccard) metric"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (intersection / union).item()

def compute_dice(pred, target, threshold=0.5):
    """Compute Dice coefficient"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    total = pred_binary.sum() + target_binary.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0

    return (2.0 * intersection / total).item()

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def load_model(model_path, device, n_filters=32, dropout=0.2):
    """Load trained model from checkpoint"""
    model = UNet(in_channels=1, n_filters=n_filters, dropout=dropout).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

def evaluate_on_test_images(model, test_images_dir, device, output_dir, tile_row=3, tile_col=4):
    """Evaluate model on test images and save visualizations"""
    test_images_dir = Path(test_images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all test images
    image_files = []
    for ext in ['*.png', '*.tif', '*.tiff', '*.jpg', '*.jpeg']:
        image_files.extend(test_images_dir.glob(ext))
    image_files = sorted(image_files)

    if len(image_files) == 0:
        print(f"❌ No test images found in {test_images_dir}")
        return

    print(f"Found {len(image_files)} test images")

    results = []

    for img_path in tqdm(image_files, desc="Evaluating"):
        # Load and preprocess image
        img = Image.open(img_path)
        img_array = np.array(img).astype(np.float32)
        if img_array.max() > 1:
            img_array /= 255.0

        # Extract center tile
        tile = extract_center_tile(img_array, tile_size=512, row=tile_row, col=tile_col)
        tile_tensor = preprocess_tile(tile)

        # Run inference
        with torch.no_grad():
            prediction = model(tile_tensor.to(device))

        pred_np = prediction.cpu().squeeze().numpy()

        # Note: We don't have ground truth masks for test_images
        # So we'll save predictions and visualizations only

        # Save visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(tile, cmap='gray')
        axes[0].set_title(f'Input Tile\n{img_path.name}')
        axes[0].axis('off')

        axes[1].imshow(pred_np, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f'Prediction\n(cell probability)')
        axes[1].axis('off')

        plt.tight_layout()
        save_path = output_dir / f"{img_path.stem}_prediction.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        # Count predicted cells (simple blob count)
        binary_pred = (pred_np > 0.5).astype(np.uint8)
        cell_count = np.sum(binary_pred) / (20 * 20)  # Rough estimate: assume 20×20px per cell

        results.append({
            'image': img_path.name,
            'dilution': img_path.stem.split('_')[0],
            'predicted_cell_count': float(cell_count),
            'mean_probability': float(pred_np.mean()),
            'max_probability': float(pred_np.max()),
        })

    # Save results
    results_path = output_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {results_path}")
    print(f"  Visualizations saved to: {output_dir}")

    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate edge detector models on test images')
    parser.add_argument('--frozen_model', type=str,
                       default='./edge_detector_experiment/20251030_122713/unet_frozen_layer1/best_model.pth',
                       help='Path to frozen layer 1 model')
    parser.add_argument('--trainable_model', type=str,
                       default='./edge_detector_experiment/20251030_122737/unet_trainable_layer1/best_model.pth',
                       help='Path to trainable layer 1 model')
    parser.add_argument('--baseline_model', type=str,
                       default='./best_models_PyTorch/unet/best_model.pth',
                       help='Path to baseline model (random init)')
    parser.add_argument('--test_images_dir', type=str, default='./test_images',
                       help='Directory with test images')
    parser.add_argument('--output_dir', type=str, default='./edge_detector_test_evaluation',
                       help='Output directory')
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--tile_row', type=int, default=3)
    parser.add_argument('--tile_col', type=int, default=4)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        'frozen_layer1': args.frozen_model,
        'trainable_layer1': args.trainable_model,
        'baseline': args.baseline_model,
    }

    all_results = {}

    for name, model_path in models.items():
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"⚠ Skipping {name}: Model not found at {model_path}")
            continue

        print(f"\n{'='*80}")
        print(f"Evaluating: {name}")
        print(f"Model: {model_path}")
        print('='*80)

        model = load_model(model_path, device, args.n_filters, args.dropout)

        model_output_dir = output_dir / name
        results = evaluate_on_test_images(
            model, args.test_images_dir, device, model_output_dir,
            tile_row=args.tile_row, tile_col=args.tile_col
        )

        all_results[name] = results

    # Create comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print('='*80)

    summary_path = output_dir / 'comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"✓ Comparison summary saved to: {summary_path}")

    # Create comparison plots
    if all_results:
        fig, ax = plt.subplots(figsize=(14, 6))

        images = [r['image'] for r in list(all_results.values())[0]]
        x = np.arange(len(images))
        width = 0.25

        for i, (name, results) in enumerate(all_results.items()):
            counts = [r['predicted_cell_count'] for r in results]
            ax.bar(x + i*width, counts, width, label=name)

        ax.set_xlabel('Test Image')
        ax.set_ylabel('Predicted Cell Count')
        ax.set_title('Edge Detector Experiment: Test Set Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels([img.split('_')[0] for img in images], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'comparison_plot.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Comparison plot saved to: {output_dir / 'comparison_plot.png'}")

if __name__ == '__main__':
    main()
