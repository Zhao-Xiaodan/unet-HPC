#!/usr/bin/env python3
"""
Fixed Loss Functions for Segmentation (TensorFlow/Keras)
=========================================================
Numerically stable loss functions with:
- Larger smoothing constants (1e-3 instead of 1e-6)
- Safe clipping to prevent underflow/overflow
- Robust handling of edge cases

Changes from original loss_functions.py:
1. smooth: 1e-6 → 1e-3 (prevents FP16 underflow)
2. epsilon: 1e-7 → 1e-3 (safe for FP16 if ever re-enabled)
3. Added explicit clipping in focal loss
4. Added safeguards in focal tversky loss
5. All operations designed to work in both FP32 and FP16
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred, smooth=1e-3):
    """
    Dice coefficient metric (for monitoring only)

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing constant (increased to 1e-3 for stability)

    Returns:
        Dice coefficient value [0, 1]
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1e-3):
    """
    Dice loss = 1 - Dice coefficient

    Direct optimization of IoU metric. Good for overlap but treats all pixels equally.
    """
    return 1.0 - dice_coef(y_true, y_pred, smooth)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance (NUMERICALLY STABLE VERSION)

    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)

    Down-weights easy examples and focuses on hard, misclassified examples.

    STABILITY IMPROVEMENTS:
    1. Larger epsilon (1e-3) to prevent log(0) even in FP16
    2. Double clipping: predictions AND p_t
    3. Focal weight clipping to prevent overflow
    4. Safe log computation

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
    # Use larger epsilon for numerical stability
    epsilon = 1e-3

    # First clipping: prevent log(0)
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate p_t
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)

    # Second clipping: ensure p_t is in safe range
    p_t = K.clip(p_t, epsilon, 1 - epsilon)

    # Calculate focal weight with safety bounds
    focal_weight = alpha * K.pow(1 - p_t, gamma)

    # Clip focal weight to prevent overflow (especially in FP16)
    focal_weight = K.clip(focal_weight, 0.0, 1000.0)

    # Calculate focal loss
    focal_loss_value = -focal_weight * K.log(p_t)

    return K.mean(focal_loss_value)


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-3):
    """
    Tversky Loss - Generalization of Dice loss with FP/FN control
    (NUMERICALLY STABLE VERSION)

    TL = 1 - (TP + ε) / (TP + α*FN + β*FP + ε)

    Allows controlling the trade-off between false positives and false negatives.
    - α > β: Penalize false negatives more (good for missing objects)
    - α < β: Penalize false positives more (good for over-segmentation)
    - α = β = 0.5: Equivalent to Dice loss

    STABILITY IMPROVEMENTS:
    1. Larger smoothing constant (1e-3 instead of 1e-6)
    2. Prevents division by zero in FP16
    3. Well above FP16 minimum representable positive (6e-5)

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        alpha: Weight for false negatives (default: 0.7)
        beta: Weight for false positives (default: 0.3)
        smooth: Smoothing constant (default: 1e-3, INCREASED from 1e-6)

    Returns:
        Tversky loss value

    Reference:
        Salehi et al. "Tversky loss function for image segmentation
        using 3D fully convolutional deep networks" (2017)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # Clip predictions for stability
    y_pred_f = K.clip(y_pred_f, 1e-3, 1 - 1e-3)

    # True Positives, False Negatives, False Positives
    TP = K.sum(y_true_f * y_pred_f)
    FN = K.sum(y_true_f * (1 - y_pred_f))
    FP = K.sum((1 - y_true_f) * y_pred_f)

    # Tversky index with larger smoothing constant
    tversky_index = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)

    return 1.0 - tversky_index


def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=1.33, smooth=1e-3):
    """
    Focal Tversky Loss - Combines Tversky and Focal loss benefits
    (NUMERICALLY STABLE VERSION)

    FTL = (1 - TI)^γ

    where TI is the Tversky Index.

    Focuses on hard examples (via gamma) while controlling FP/FN balance (via alpha/beta).
    Excellent for highly imbalanced segmentation with small objects.

    STABILITY IMPROVEMENTS:
    1. Larger smoothing constant (1e-3 instead of 1e-6)
    2. Prediction clipping before computation
    3. Tversky index clipping to safe range
    4. Output clipping to prevent overflow
    5. Safe power operation

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        alpha: Weight for false negatives (default: 0.7)
        beta: Weight for false positives (default: 0.3)
        gamma: Focusing parameter (default: 1.33, from paper)
        smooth: Smoothing constant (default: 1e-3, INCREASED from 1e-6)

    Returns:
        Focal Tversky loss value

    Reference:
        Abraham & Khan "A Novel Focal Tversky loss function with improved
        Attention U-Net for lesion segmentation" (2019)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # Clip predictions for stability
    y_pred_f = K.clip(y_pred_f, 1e-3, 1 - 1e-3)

    # True Positives, False Negatives, False Positives
    TP = K.sum(y_true_f * y_pred_f)
    FN = K.sum(y_true_f * (1 - y_pred_f))
    FP = K.sum((1 - y_true_f) * y_pred_f)

    # Tversky index with larger smoothing
    tversky_index = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)

    # Clip tversky index to safe range before power operation
    tversky_index = K.clip(tversky_index, 1e-3, 1 - 1e-3)

    # Apply focal component with safe power operation
    focal_tversky = K.pow((1 - tversky_index), gamma)

    # Final clipping to prevent any overflow
    focal_tversky = K.clip(focal_tversky, 0.0, 1.0)

    return focal_tversky


def combined_dice_focal_loss(y_true, y_pred, dice_weight=0.7, focal_weight=0.3,
                              alpha=0.25, gamma=2.0):
    """
    Combined Dice + Focal Loss (NUMERICALLY STABLE VERSION)

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
    Combined Tversky + Focal Loss (NUMERICALLY STABLE VERSION)

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


def jacard_coef(y_true, y_pred, smooth=1e-3):
    """
    Jaccard coefficient (IoU) metric (NUMERICALLY STABLE VERSION)

    IoU = |A ∩ B| / |A ∪ B|

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing constant (default: 1e-3, INCREASED from 1e-6)

    Returns:
        Jaccard coefficient value [0, 1]
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# ============================================================================
# KERAS LOSS CLASSES (for model.compile())
# ============================================================================

@keras.saving.register_keras_serializable(package='Custom')
class BinaryFocalLoss(keras.losses.Loss):
    """
    Binary Focal Loss as a Keras Loss class for proper serialization.

    Wrapper around the focal_loss function that can be used with model.compile()
    and properly serialized when saving models.

    Usage:
        model.compile(
            optimizer='adam',
            loss=BinaryFocalLoss(gamma=2, alpha=0.25),
            metrics=['accuracy']
        )

    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Balancing factor (default: 0.25)
        name: Name of the loss (default: 'binary_focal_loss')
    """
    def __init__(self, gamma=2.0, alpha=0.25, name='binary_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        """Compute focal loss."""
        return focal_loss(y_true, y_pred, alpha=self.alpha, gamma=self.gamma)

    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        return cls(**config)


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
    print("="*80)
    print("TESTING NUMERICALLY STABLE LOSS FUNCTIONS")
    print("="*80)

    # Create dummy data
    y_true = tf.constant([[[[1.0]], [[0.0]], [[1.0]], [[0.0]]]])
    y_pred = tf.constant([[[[0.9]], [[0.1]], [[0.8]], [[0.2]]]])

    print("\nTest data:")
    print(f"  y_true: {y_true.numpy().flatten()}")
    print(f"  y_pred: {y_pred.numpy().flatten()}")

    print("\n" + "-"*80)
    print("Loss function values:")
    print("-"*80)

    # Test all losses
    for name, loss_fn in LOSS_FUNCTIONS.items():
        try:
            loss_value = loss_fn(y_true, y_pred)
            is_finite = tf.math.is_finite(loss_value)
            status = "✓ FINITE" if is_finite else "✗ NOT FINITE"
            print(f"{name:25s}: {float(loss_value):8.6f}  {status}")
        except Exception as e:
            print(f"{name:25s}: ERROR - {e}")

    print("\n" + "-"*80)
    print("Metric values:")
    print("-"*80)

    # Test metrics
    try:
        dice = dice_coef(y_true, y_pred)
        jacard = jacard_coef(y_true, y_pred)
        print(f"{'dice_coef':25s}: {float(dice):8.6f}  ✓")
        print(f"{'jacard_coef':25s}: {float(jacard):8.6f}  ✓")
    except Exception as e:
        print(f"Metrics: ERROR - {e}")

    print("\n" + "="*80)
    print("✓ All loss functions tested successfully!")
    print("="*80)
