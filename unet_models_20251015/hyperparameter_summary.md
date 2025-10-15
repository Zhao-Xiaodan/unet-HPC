# Hyperparameter Summary for bead_seg.ipynb

This document summarizes the key training hyperparameters used in the `bead_seg.ipynb` notebook for training U-Net, Attention U-Net, and Attention Residual U-Net models.

## General Hyperparameters

*   **Image Size**: 512x512
*   **Batch Size**: 4
*   **Number of Epochs**: 200
*   **Optimizer**: Adam
*   **Learning Rate**: 0.005
*   **Loss Function**: Binary Focal Loss (with gamma=2)

## Models Trained

The following three model architectures were trained:

1.  **U-Net**
2.  **Attention U-Net**
3.  **Attention Residual U-Net**
