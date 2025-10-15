#!/usr/bin/env python3
"""
Wrapper script to run dataset size study with proper dependency handling.
"""

import os
import sys
import subprocess

def install_focal_loss():
    """Install focal_loss if not available"""
    try:
        from focal_loss import BinaryFocalLoss
        print("✓ focal_loss already available")
        return True
    except ImportError:
        print("Installing focal_loss...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "focal-loss"])
            print("✓ focal_loss installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install focal_loss: {e}")
            print("Creating custom focal_loss implementation...")
            create_custom_focal_loss()
            return True

def create_custom_focal_loss():
    """Create custom focal loss implementation if package not available"""
    with open('focal_loss.py', 'w') as f:
        f.write('''
import tensorflow as tf
from tensorflow.keras import backend as K

class BinaryFocalLoss:
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        """
        Binary focal loss implementation
        """
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Calculate focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)

        focal_loss = -alpha_t * K.pow(1 - pt, self.gamma) * K.log(pt)

        return K.mean(focal_loss)
''')

def run_dataset_percentage(percentage):
    """Run training for a specific dataset percentage"""
    print(f"\n{'='*60}")
    print(f"🔬 RUNNING DATASET SIZE STUDY: {percentage}% OF FULL DATASET")
    print(f"{'='*60}")

    # Set environment variable for dataset percentage
    os.environ['DATASET_PERCENTAGE'] = str(percentage)

    # Import and run the study script
    try:
        exec(open('224_225_226_dataset_size_study.py').read())
        print(f"✅ Completed study for {percentage}% dataset")
        return True
    except Exception as e:
        print(f"❌ Failed study for {percentage}% dataset: {e}")
        return False

def main():
    print("=== MITOCHONDRIA DATASET SIZE SUFFICIENCY STUDY ===")
    print("🔬 Testing multiple dataset sizes for acceptable performance")
    print()

    # Install focal loss
    install_focal_loss()

    # Dataset percentages to test
    percentages = [10, 20, 50, 75, 100]

    results_summary = {}

    print(f"\n🎯 DATASET SIZE STUDY PLAN:")
    print("-" * 30)
    for pct in percentages:
        estimated_samples = int(1980 * pct / 100)
        print(f"  {pct:3d}%: ~{estimated_samples:4d} samples")
    print()

    # Run study for each percentage
    successful_runs = 0
    for percentage in percentages:
        try:
            success = run_dataset_percentage(percentage)
            if success:
                successful_runs += 1
                results_summary[f'{percentage}%'] = 'Completed'
            else:
                results_summary[f'{percentage}%'] = 'Failed'
        except Exception as e:
            print(f"❌ Critical error for {percentage}%: {e}")
            results_summary[f'{percentage}%'] = f'Error: {e}'

    # Final summary
    print(f"\n{'='*60}")
    print("🏁 DATASET SIZE STUDY COMPLETED")
    print(f"{'='*60}")
    print(f"✅ Successful runs: {successful_runs}/{len(percentages)}")
    print("\n📋 RESULTS SUMMARY:")
    for pct, status in results_summary.items():
        print(f"  {pct:4s}: {status}")

    print(f"\n📁 Individual results saved in respective output directories")
    print("🔍 Check each directory for detailed analysis and visualizations")

if __name__ == "__main__":
    main()
