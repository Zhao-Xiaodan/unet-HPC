#!/usr/bin/env python3
"""
Advanced Loss Functions for Segmentation (TensorFlow/Keras)
============================================================
Collection of advanced loss functions for microbead segmentation including:
- Focal Loss
- Tversky Loss
- Focal Tversky Loss
- Combined losses with configurable weights
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient metric (for monitoring only)

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing constant to avoid division by zero

    Returns:
        Dice coefficient value [0, 1]
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss = 1 - Dice coefficient

    Direct optimization of IoU metric. Good for overlap but treats all pixels equally.
    """
    return 1.0 - dice_coef(y_true, y_pred, smooth)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance

    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)

    Down-weights easy examples and focuses on hard, misclassified examples.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        alpha: Balancing factor for positive/negative classes (default: 0.25)
        gamma: Focusing parameter, higher = more focus on hard examples (default: 2.0)

    Returns:
        Focal loss value

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    # Clip predictions to prevent log(0)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    # Calculate p_t
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)

    # Calculate focal weight
    focal_weight = alpha * K.pow(1 - p_t, gamma)

    # Calculate focal loss
    focal_loss_value = -focal_weight * K.log(p_t)

    return K.mean(focal_loss_value)


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    """
    Tversky Loss - Generalization of Dice loss with FP/FN control

    TL = 1 - (TP + ε) / (TP + α*FN + β*FP + ε)

    Allows controlling the trade-off between false positives and false negatives.
    - α > β: Penalize false negatives more (good for missing objects)
    - α < β: Penalize false positives more (good for over-segmentation)
    - α = β = 0.5: Equivalent to Dice loss

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        alpha: Weight for false negatives (default: 0.7)
        beta: Weight for false positives (default: 0.3)
        smooth: Smoothing constant

    Returns:
        Tversky loss value

    Reference:
        Salehi et al. "Tversky loss function for image segmentation
        using 3D fully convolutional deep networks" (2017)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # True Positives, False Negatives, False Positives
    TP = K.sum(y_true_f * y_pred_f)
    FN = K.sum(y_true_f * (1 - y_pred_f))
    FP = K.sum((1 - y_true_f) * y_pred_f)

    tversky_index = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)

    return 1.0 - tversky_index


def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=1.33, smooth=1e-6):
    """
    Focal Tversky Loss - Combines Tversky and Focal loss benefits

    FTL = (1 - TI)^γ

    where TI is the Tversky Index.

    Focuses on hard examples (via gamma) while controlling FP/FN balance (via alpha/beta).
    Excellent for highly imbalanced segmentation with small objects.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        alpha: Weight for false negatives (default: 0.7)
        beta: Weight for false positives (default: 0.3)
        gamma: Focusing parameter (default: 1.33, from paper)
        smooth: Smoothing constant

    Returns:
        Focal Tversky loss value

    Reference:
        Abraham & Khan "A Novel Focal Tversky loss function with improved
        Attention U-Net for lesion segmentation" (2019)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # True Positives, False Negatives, False Positives
    TP = K.sum(y_true_f * y_pred_f)
    FN = K.sum(y_true_f * (1 - y_pred_f))
    FP = K.sum((1 - y_true_f) * y_pred_f)

    tversky_index = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)

    # Apply focal component
    focal_tversky = K.pow((1 - tversky_index), gamma)

    return focal_tversky


def combined_dice_focal_loss(y_true, y_pred, dice_weight=0.7, focal_weight=0.3,
                              alpha=0.25, gamma=2.0):
    """
    Combined Dice + Focal Loss

    L_combined = w_dice * L_dice + w_focal * L_focal

    Balances global overlap optimization (Dice) with hard example mining (Focal).
    Recommended from previous hyperparameter search results.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        dice_weight: Weight for dice loss (default: 0.7)
        focal_weight: Weight for focal loss (default: 0.3)
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter

    Returns:
        Combined loss value
    """
    return dice_weight * dice_loss(y_true, y_pred) + \
           focal_weight * focal_loss(y_true, y_pred, alpha, gamma)


def combined_tversky_focal_loss(y_true, y_pred, tversky_weight=0.6, focal_weight=0.4,
                                 alpha=0.7, beta=0.3, gamma=2.0):
    """
    Combined Tversky + Focal Loss

    L_combined = w_tversky * L_tversky + w_focal * L_focal

    Combines FP/FN control (Tversky) with hard example mining (Focal).
    Good for imbalanced datasets with touching objects.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        tversky_weight: Weight for Tversky loss (default: 0.6)
        focal_weight: Weight for focal loss (default: 0.4)
        alpha: Tversky alpha parameter (FN weight)
        beta: Tversky beta parameter (FP weight)
        gamma: Focal loss gamma parameter

    Returns:
        Combined loss value
    """
    return tversky_weight * tversky_loss(y_true, y_pred, alpha, beta) + \
           focal_weight * focal_loss(y_true, y_pred, alpha=0.25, gamma=gamma)


def jacard_coef(y_true, y_pred, smooth=1e-6):
    """
    Jaccard coefficient (IoU) metric

    IoU = |A ∩ B| / |A ∪ B|

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing constant

    Returns:
        Jaccard coefficient value [0, 1]
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# Dictionary of available loss functions
LOSS_FUNCTIONS = {
    'focal': focal_loss,
    'tversky': tversky_loss,
    'focal_tversky': focal_tversky_loss,
    'combined': combined_dice_focal_loss,
    'combined_tversky': combined_tversky_focal_loss,
}


def get_loss_function(loss_name):
    """
    Get loss function by name

    Args:
        loss_name: Name of loss function (see LOSS_FUNCTIONS dict)

    Returns:
        Loss function

    Raises:
        ValueError: If loss_name not found
    """
    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Available: {list(LOSS_FUNCTIONS.keys())}")

    return LOSS_FUNCTIONS[loss_name]


if __name__ == '__main__':
    # Test loss functions
    print("Testing loss functions...")
    print("-" * 80)

    # Create dummy data
    y_true = tf.constant([[[[1.0]], [[0.0]], [[1.0]], [[0.0]]]])
    y_pred = tf.constant([[[[0.9]], [[0.1]], [[0.8]], [[0.2]]]])

    # Test all losses
    for name, loss_fn in LOSS_FUNCTIONS.items():
        try:
            loss_value = loss_fn(y_true, y_pred)
            print(f"{name:25s}: {float(loss_value):.6f}")
        except Exception as e:
            print(f"{name:25s}: ERROR - {e}")

    print("-" * 80)
    print("✓ Loss functions tested successfully!")
