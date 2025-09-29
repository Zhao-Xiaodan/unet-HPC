#!/usr/bin/env python3
"""
Wrapper script to run ORIGINAL BROKEN mitochondria segmentation
with proper dependency handling.
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

def main():
    print("=== ORIGINAL BROKEN MITOCHONDRIA SEGMENTATION WRAPPER ===")
    print("🚨 WARNING: This uses the BROKEN Jaccard implementation!")
    print()

    # Install focal loss
    install_focal_loss()

    print()
    print("🚨 Starting ORIGINAL BROKEN implementation training...")
    print("=" * 60)

    # Import and run the main training script
    exec(open('224_225_226_mito_segm_using_various_unet_models_original.py').read())

if __name__ == "__main__":
    main()
