#!/usr/bin/env python3
"""
Batch Feature Visualization for Edge Detector Experiment
=========================================================

Generate feature visualizations for all three models:
1. Baseline (random init)
2. Frozen layer 1 (Gabor init, frozen)
3. Trainable layer 1 (Gabor init, trainable)

This enables direct comparison of layer 1 features across initialization strategies.

Author: Claude Code
Date: October 30, 2025
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import argparse


def find_latest_experiment_dir(base_dir='./edge_detector_experiment'):
    """Find the latest experiment directory by timestamp"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None

    # Find all timestamp directories
    timestamp_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        return None

    # Sort by name (timestamp format ensures chronological order)
    latest_dir = sorted(timestamp_dirs)[-1]
    return latest_dir


def run_visualization(model_path, output_dir, n_filters=32, dropout=0.2,
                     layers=None, channels_per_layer=12, diverse_per_channel=3,
                     iterations=500):
    """
    Run feature visualization for a single model

    Args:
        model_path: Path to model checkpoint
        output_dir: Output directory for visualizations
        n_filters: Number of base filters
        dropout: Dropout rate
        layers: List of layer names to visualize
        channels_per_layer: Number of channels per layer
        diverse_per_channel: Number of diverse examples per channel
        iterations: Number of optimization iterations
    """
    if layers is None:
        layers = [
            'encoder_1_conv2',  # <- KEY LAYER for comparison
            'encoder_2_conv2',
            'encoder_3_conv2',
            'bottleneck_conv2',
            'decoder_3_conv2',
            'decoder_1_conv2',
        ]

    # Build command
    cmd = [
        'python', 'unet_feature_visualization.py',
        '--model_path', str(model_path),
        '--output_dir', str(output_dir),
        '--n_filters', str(n_filters),
        '--dropout', str(dropout),
        '--layers', *layers,
        '--channels_per_layer', str(channels_per_layer),
        '--diverse_per_channel', str(diverse_per_channel),
        '--iterations', str(iterations),
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Run visualization
    result = subprocess.run(cmd, check=True)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Batch feature visualization for edge detector experiment'
    )
    parser.add_argument('--experiment_dir', type=str,
                       default='./edge_detector_experiment',
                       help='Base experiment directory')
    parser.add_argument('--baseline_model', type=str,
                       default='./best_models_PyTorch/unet/best_model.pth',
                       help='Path to baseline model (random init)')
    parser.add_argument('--baseline_viz_dir', type=str,
                       default='./unet_feature_viz_20251029_065244',
                       help='Existing baseline visualization directory (skip if exists)')
    parser.add_argument('--n_filters', type=int, default=32,
                       help='Number of base filters')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--channels_per_layer', type=int, default=12,
                       help='Channels to visualize per layer')
    parser.add_argument('--diverse_per_channel', type=int, default=3,
                       help='Diverse examples per channel')
    parser.add_argument('--iterations', type=int, default=500,
                       help='Optimization iterations')
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip baseline visualization (already exists)')

    args = parser.parse_args()

    print("="*70)
    print("Edge Detector Experiment: Batch Feature Visualization")
    print("="*70)

    # Find latest experiment directory
    exp_dir = find_latest_experiment_dir(args.experiment_dir)
    if exp_dir is None:
        print(f"\n❌ ERROR: No experiment directory found in {args.experiment_dir}")
        print("Please run training first:")
        print("  qsub pbs_edge_detector_frozen.sh")
        print("  qsub pbs_edge_detector_trainable.sh")
        sys.exit(1)

    print(f"\nExperiment directory: {exp_dir}")

    # Define models to visualize
    models_to_viz = []

    # 1. Baseline (if requested)
    if not args.skip_baseline:
        baseline_path = Path(args.baseline_model)
        if baseline_path.exists():
            models_to_viz.append({
                'name': 'Baseline (Random Init)',
                'model_path': baseline_path,
                'output_dir': exp_dir / 'visualizations' / 'baseline_random_init',
                'description': 'Standard U-Net with random initialization (IoU=0.6377)'
            })
        else:
            print(f"\n⚠️  Baseline model not found: {baseline_path}")
    else:
        print(f"\n✓ Skipping baseline (using existing: {args.baseline_viz_dir})")

    # 2. Frozen layer 1
    frozen_model = exp_dir / 'unet_frozen_layer1' / 'best_model.pth'
    if frozen_model.exists():
        models_to_viz.append({
            'name': 'Frozen Layer 1',
            'model_path': frozen_model,
            'output_dir': exp_dir / 'visualizations' / 'frozen_layer1',
            'description': 'Gabor filters frozen throughout training'
        })
    else:
        print(f"\n⚠️  Frozen model not found: {frozen_model}")
        print("Run: qsub pbs_edge_detector_frozen.sh")

    # 3. Trainable layer 1
    trainable_model = exp_dir / 'unet_trainable_layer1' / 'best_model.pth'
    if trainable_model.exists():
        models_to_viz.append({
            'name': 'Trainable Layer 1',
            'model_path': trainable_model,
            'output_dir': exp_dir / 'visualizations' / 'trainable_layer1',
            'description': 'Gabor filters allowed to adapt during training'
        })
    else:
        print(f"\n⚠️  Trainable model not found: {trainable_model}")
        print("Run: qsub pbs_edge_detector_trainable.sh")

    if len(models_to_viz) == 0:
        print("\n❌ ERROR: No models found to visualize!")
        sys.exit(1)

    # Run visualizations
    print(f"\nFound {len(models_to_viz)} model(s) to visualize")
    print("="*70)

    results = []
    for i, model_info in enumerate(models_to_viz, 1):
        print(f"\n[{i}/{len(models_to_viz)}] {model_info['name']}")
        print(f"Description: {model_info['description']}")
        print(f"Model: {model_info['model_path']}")
        print(f"Output: {model_info['output_dir']}")

        # Check if visualization already exists
        if model_info['output_dir'].exists():
            print(f"\n⚠️  Output directory already exists: {model_info['output_dir']}")
            response = input("Overwrite? [y/N]: ")
            if response.lower() != 'y':
                print("Skipping...")
                results.append({**model_info, 'status': 'skipped'})
                continue

        try:
            success = run_visualization(
                model_path=model_info['model_path'],
                output_dir=model_info['output_dir'],
                n_filters=args.n_filters,
                dropout=args.dropout,
                channels_per_layer=args.channels_per_layer,
                diverse_per_channel=args.diverse_per_channel,
                iterations=args.iterations,
            )

            if success:
                print(f"\n✓ Visualization complete: {model_info['output_dir']}")
                results.append({**model_info, 'status': 'success'})
            else:
                print(f"\n❌ Visualization failed for {model_info['name']}")
                results.append({**model_info, 'status': 'failed'})

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            results.append({**model_info, 'status': 'error', 'error': str(e)})

        print("\n" + "="*70)

    # Summary
    print("\nBatch Visualization Summary:")
    print("="*70)

    for result in results:
        status_symbol = {
            'success': '✓',
            'skipped': '⊘',
            'failed': '❌',
            'error': '❌',
        }.get(result['status'], '?')

        print(f"{status_symbol} {result['name']}: {result['status']}")
        if result['status'] == 'success':
            print(f"  → {result['output_dir']}")

    print("="*70)

    # Next steps
    print("\nNext steps:")
    print("1. Compare encoder_1_conv2 visualizations:")
    if not args.skip_baseline:
        print(f"   - Baseline: {exp_dir}/visualizations/baseline_random_init/encoder_1_conv2/")
    else:
        print(f"   - Baseline: {args.baseline_viz_dir}/encoder_1_conv2/")
    print(f"   - Frozen: {exp_dir}/visualizations/frozen_layer1/encoder_1_conv2/")
    print(f"   - Trainable: {exp_dir}/visualizations/trainable_layer1/encoder_1_conv2/")
    print()
    print("2. Check if Gabor filters adapted (trainable variant):")
    print(f"   python -c \"import torch; w0=torch.load('{exp_dir}/unet_trainable_layer1/layer1_weights_epoch000.pth'); wf=torch.load('{exp_dir}/unet_trainable_layer1/layer1_weights_final.pth'); print('L2 distance:', torch.norm(wf-w0).item())\"")
    print()
    print("3. Compare performance:")
    print(f"   cat {exp_dir}/unet_frozen_layer1/model_info.json")
    print(f"   cat {exp_dir}/unet_trainable_layer1/model_info.json")
    print()
    print("4. Create comparative analysis report:")
    print("   python analyze_edge_detector_experiment.py")
    print()

    # Save summary
    summary_path = exp_dir / 'visualization_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'experiment_dir': str(exp_dir),
            'visualizations': [
                {
                    'name': r['name'],
                    'model_path': str(r['model_path']),
                    'output_dir': str(r['output_dir']),
                    'status': r['status'],
                }
                for r in results
            ],
        }, f, indent=2)

    print(f"✓ Summary saved to: {summary_path}")
    print()


if __name__ == '__main__':
    main()
