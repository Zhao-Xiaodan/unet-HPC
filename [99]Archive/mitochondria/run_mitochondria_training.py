#!/usr/bin/env python3
"""
Wrapper script to run mitochondria segmentation with proper dataset paths
and dependency handling.
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
            print("Attempting alternative installation...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "focal-loss-tensorflow"])
                print("✓ focal-loss-tensorflow installed successfully")
                return True
            except subprocess.CalledProcessError:
                print("✗ Could not install focal_loss. Will implement custom version.")
                return False

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

def fix_dataset_paths():
    """Create symbolic links to fix dataset path mismatch"""
    if not os.path.exists('data'):
        if os.path.exists('dataset'):
            print("Creating symbolic link: data -> dataset")
            os.symlink('dataset', 'data')
        else:
            print("ERROR: Neither 'data' nor 'dataset' directory found!")
            return False
    return True

def main():
    print("=== MITOCHONDRIA SEGMENTATION TRAINING WRAPPER ===")
    print()

    # Fix dataset paths
    if not fix_dataset_paths():
        sys.exit(1)

    # Install or create focal loss
    if not install_focal_loss():
        create_custom_focal_loss()
        print("✓ Created custom focal_loss implementation")

    print()
    print("Starting mitochondria segmentation training...")
    print("=" * 50)

    # Import and run the main training script
    exec(open('224_225_226_mito_segm_using_various_unet_models.py').read())

if __name__ == "__main__":
    main()
