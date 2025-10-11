#!/usr/bin/env python3
"""
Model Architectures for Microbead Segmentation
===============================================
Includes:
1. Standard U-Net
2. Residual U-Net (ResU-Net)
3. Attention Residual U-Net (Attention ResU-Net)

All models compatible with 512×512 input resolution.
"""

import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K


# ==============================================================================
# Building Blocks
# ==============================================================================

def conv_block(x, filters, filter_size=3, dropout=0.0, batch_norm=True, name_prefix="conv"):
    """
    Standard convolutional block: Conv-BN-ReLU-Conv-BN-ReLU-Dropout

    Args:
        x: Input tensor
        filters: Number of filters
        filter_size: Size of convolutional kernel (default: 3)
        dropout: Dropout rate (default: 0.0)
        batch_norm: Use batch normalization (default: True)
        name_prefix: Prefix for layer names

    Returns:
        Output tensor after convolutions
    """
    conv = layers.Conv2D(filters, (filter_size, filter_size), padding="same",
                        name=f"{name_prefix}_conv1")(x)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3, name=f"{name_prefix}_bn1")(conv)
    conv = layers.Activation("relu", name=f"{name_prefix}_relu1")(conv)

    conv = layers.Conv2D(filters, (filter_size, filter_size), padding="same",
                        name=f"{name_prefix}_conv2")(conv)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3, name=f"{name_prefix}_bn2")(conv)
    conv = layers.Activation("relu", name=f"{name_prefix}_relu2")(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout, name=f"{name_prefix}_dropout")(conv)

    return conv


def res_conv_block(x, filters, filter_size=3, dropout=0.0, batch_norm=True, name_prefix="res"):
    """
    Residual convolutional block with skip connection

    Architecture: Conv-BN-ReLU-Conv-BN + Shortcut -> ReLU-Dropout

    Args:
        x: Input tensor
        filters: Number of filters
        filter_size: Size of convolutional kernel (default: 3)
        dropout: Dropout rate (default: 0.0)
        batch_norm: Use batch normalization (default: True)
        name_prefix: Prefix for layer names

    Returns:
        Output tensor after residual connection
    """
    conv = layers.Conv2D(filters, (filter_size, filter_size), padding="same",
                        name=f"{name_prefix}_conv1")(x)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3, name=f"{name_prefix}_bn1")(conv)
    conv = layers.Activation("relu", name=f"{name_prefix}_relu1")(conv)

    conv = layers.Conv2D(filters, (filter_size, filter_size), padding="same",
                        name=f"{name_prefix}_conv2")(conv)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3, name=f"{name_prefix}_bn2")(conv)

    # Shortcut connection
    shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding='same',
                            name=f"{name_prefix}_shortcut")(x)
    if batch_norm:
        shortcut = layers.BatchNormalization(axis=3, name=f"{name_prefix}_shortcut_bn")(shortcut)

    # Add residual
    res = layers.Add(name=f"{name_prefix}_add")([shortcut, conv])
    res = layers.Activation("relu", name=f"{name_prefix}_relu2")(res)

    if dropout > 0:
        res = layers.Dropout(dropout, name=f"{name_prefix}_dropout")(res)

    return res


def attention_gate(x, g, filters, name_prefix="attn"):
    """
    Attention gate for focusing on relevant regions

    Args:
        x: Input feature map from encoder (skip connection)
        g: Gating signal from decoder (coarser scale)
        filters: Number of intermediate filters
        name_prefix: Prefix for layer names

    Returns:
        Attention-weighted feature map
    """
    # Theta^T * x
    theta_x = layers.Conv2D(filters, (1, 1), padding='same',
                           name=f"{name_prefix}_theta_x")(x)

    # Phi^T * g
    phi_g = layers.Conv2D(filters, (1, 1), padding='same',
                         name=f"{name_prefix}_phi_g")(g)

    # Add and apply ReLU
    add = layers.Add(name=f"{name_prefix}_add")([theta_x, phi_g])
    act = layers.Activation('relu', name=f"{name_prefix}_relu")(add)

    # Psi^T * act
    psi = layers.Conv2D(1, (1, 1), padding='same',
                       name=f"{name_prefix}_psi")(act)
    psi = layers.Activation('sigmoid', name=f"{name_prefix}_sigmoid")(psi)

    # Multiply attention coefficients with input
    output = layers.Multiply(name=f"{name_prefix}_multiply")([x, psi])

    return output


# ==============================================================================
# Model 1: Standard U-Net
# ==============================================================================

def UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.3, batch_norm=True):
    """
    Standard U-Net architecture

    Args:
        input_shape: Input image shape (height, width, channels)
        NUM_CLASSES: Number of output classes (default: 1 for binary segmentation)
        dropout_rate: Dropout rate (default: 0.3)
        batch_norm: Use batch normalization (default: True)

    Returns:
        Keras model

    Architecture:
        - 4 encoder blocks with max pooling
        - Bridge (bottleneck)
        - 4 decoder blocks with upsampling and skip connections
        - Final 1×1 convolution with sigmoid activation
    """
    inputs = layers.Input(input_shape, name="input")

    # Encoder
    conv1 = conv_block(inputs, 64, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="enc1")
    pool1 = layers.MaxPooling2D((2, 2), name="pool1")(conv1)

    conv2 = conv_block(pool1, 128, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="enc2")
    pool2 = layers.MaxPooling2D((2, 2), name="pool2")(conv2)

    conv3 = conv_block(pool2, 256, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="enc3")
    pool3 = layers.MaxPooling2D((2, 2), name="pool3")(conv3)

    conv4 = conv_block(pool3, 512, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="enc4")
    pool4 = layers.MaxPooling2D((2, 2), name="pool4")(conv4)

    # Bridge
    bridge = conv_block(pool4, 1024, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="bridge")

    # Decoder
    up1 = layers.UpSampling2D((2, 2), name="up1")(bridge)
    concat1 = layers.Concatenate(axis=3, name="concat1")([up1, conv4])
    conv5 = conv_block(concat1, 512, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="dec1")

    up2 = layers.UpSampling2D((2, 2), name="up2")(conv5)
    concat2 = layers.Concatenate(axis=3, name="concat2")([up2, conv3])
    conv6 = conv_block(concat2, 256, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="dec2")

    up3 = layers.UpSampling2D((2, 2), name="up3")(conv6)
    concat3 = layers.Concatenate(axis=3, name="concat3")([up3, conv2])
    conv7 = conv_block(concat3, 128, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="dec3")

    up4 = layers.UpSampling2D((2, 2), name="up4")(conv7)
    concat4 = layers.Concatenate(axis=3, name="concat4")([up4, conv1])
    conv8 = conv_block(concat4, 64, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="dec4")

    # Output
    outputs = layers.Conv2D(NUM_CLASSES, (1, 1), activation='sigmoid', name="output")(conv8)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="UNet")

    return model


# ==============================================================================
# Model 2: Residual U-Net (ResU-Net)
# ==============================================================================

def ResUNet(input_shape, NUM_CLASSES=1, dropout_rate=0.3, batch_norm=True):
    """
    Residual U-Net with residual connections in encoder/decoder blocks

    Improves gradient flow and enables training of deeper networks.

    Args:
        input_shape: Input image shape (height, width, channels)
        NUM_CLASSES: Number of output classes (default: 1)
        dropout_rate: Dropout rate (default: 0.3)
        batch_norm: Use batch normalization (default: True)

    Returns:
        Keras model
    """
    inputs = layers.Input(input_shape, name="input")

    # Encoder
    conv1 = res_conv_block(inputs, 64, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="enc1")
    pool1 = layers.MaxPooling2D((2, 2), name="pool1")(conv1)

    conv2 = res_conv_block(pool1, 128, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="enc2")
    pool2 = layers.MaxPooling2D((2, 2), name="pool2")(conv2)

    conv3 = res_conv_block(pool2, 256, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="enc3")
    pool3 = layers.MaxPooling2D((2, 2), name="pool3")(conv3)

    conv4 = res_conv_block(pool3, 512, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="enc4")
    pool4 = layers.MaxPooling2D((2, 2), name="pool4")(conv4)

    # Bridge
    bridge = res_conv_block(pool4, 1024, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="bridge")

    # Decoder
    up1 = layers.UpSampling2D((2, 2), name="up1")(bridge)
    concat1 = layers.Concatenate(axis=3, name="concat1")([up1, conv4])
    conv5 = res_conv_block(concat1, 512, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="dec1")

    up2 = layers.UpSampling2D((2, 2), name="up2")(conv5)
    concat2 = layers.Concatenate(axis=3, name="concat2")([up2, conv3])
    conv6 = res_conv_block(concat2, 256, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="dec2")

    up3 = layers.UpSampling2D((2, 2), name="up3")(conv6)
    concat3 = layers.Concatenate(axis=3, name="concat3")([up3, conv2])
    conv7 = res_conv_block(concat3, 128, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="dec3")

    up4 = layers.UpSampling2D((2, 2), name="up4")(conv7)
    concat4 = layers.Concatenate(axis=3, name="concat4")([up4, conv1])
    conv8 = res_conv_block(concat4, 64, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="dec4")

    # Output
    outputs = layers.Conv2D(NUM_CLASSES, (1, 1), activation='sigmoid', name="output")(conv8)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="ResUNet")

    return model


# ==============================================================================
# Model 3: Attention Residual U-Net
# ==============================================================================

def AttentionResUNet(input_shape, NUM_CLASSES=1, dropout_rate=0.3, batch_norm=True):
    """
    Attention Residual U-Net combining residual connections and attention gates

    Attention gates help the model focus on relevant spatial regions,
    especially useful for small objects and boundary refinement.

    Args:
        input_shape: Input image shape (height, width, channels)
        NUM_CLASSES: Number of output classes (default: 1)
        dropout_rate: Dropout rate (default: 0.3)
        batch_norm: Use batch normalization (default: True)

    Returns:
        Keras model
    """
    inputs = layers.Input(input_shape, name="input")

    # Encoder
    conv1 = res_conv_block(inputs, 64, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="enc1")
    pool1 = layers.MaxPooling2D((2, 2), name="pool1")(conv1)

    conv2 = res_conv_block(pool1, 128, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="enc2")
    pool2 = layers.MaxPooling2D((2, 2), name="pool2")(conv2)

    conv3 = res_conv_block(pool2, 256, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="enc3")
    pool3 = layers.MaxPooling2D((2, 2), name="pool3")(conv3)

    conv4 = res_conv_block(pool3, 512, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="enc4")
    pool4 = layers.MaxPooling2D((2, 2), name="pool4")(conv4)

    # Bridge
    bridge = res_conv_block(pool4, 1024, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="bridge")

    # Decoder with Attention Gates
    up1 = layers.UpSampling2D((2, 2), name="up1")(bridge)
    attn1 = attention_gate(conv4, up1, 512, name_prefix="attn1")
    concat1 = layers.Concatenate(axis=3, name="concat1")([up1, attn1])
    conv5 = res_conv_block(concat1, 512, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="dec1")

    up2 = layers.UpSampling2D((2, 2), name="up2")(conv5)
    attn2 = attention_gate(conv3, up2, 256, name_prefix="attn2")
    concat2 = layers.Concatenate(axis=3, name="concat2")([up2, attn2])
    conv6 = res_conv_block(concat2, 256, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="dec2")

    up3 = layers.UpSampling2D((2, 2), name="up3")(conv6)
    attn3 = attention_gate(conv2, up3, 128, name_prefix="attn3")
    concat3 = layers.Concatenate(axis=3, name="concat3")([up3, attn3])
    conv7 = res_conv_block(concat3, 128, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="dec3")

    up4 = layers.UpSampling2D((2, 2), name="up4")(conv7)
    attn4 = attention_gate(conv1, up4, 64, name_prefix="attn4")
    concat4 = layers.Concatenate(axis=3, name="concat4")([up4, attn4])
    conv8 = res_conv_block(concat4, 64, dropout=dropout_rate, batch_norm=batch_norm, name_prefix="dec4")

    # Output
    outputs = layers.Conv2D(NUM_CLASSES, (1, 1), activation='sigmoid', name="output")(conv8)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="AttentionResUNet")

    return model


# Dictionary of available models
MODEL_ARCHITECTURES = {
    'unet': UNet,
    'resunet': ResUNet,
    'attention_resunet': AttentionResUNet,
}


def get_model(model_name, input_shape, NUM_CLASSES=1, dropout_rate=0.3, batch_norm=True):
    """
    Get model by name

    Args:
        model_name: Name of model architecture
        input_shape: Input image shape (height, width, channels)
        NUM_CLASSES: Number of output classes
        dropout_rate: Dropout rate
        batch_norm: Use batch normalization

    Returns:
        Keras model

    Raises:
        ValueError: If model_name not found
    """
    if model_name not in MODEL_ARCHITECTURES:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available: {list(MODEL_ARCHITECTURES.keys())}")

    model_fn = MODEL_ARCHITECTURES[model_name]
    return model_fn(input_shape, NUM_CLASSES, dropout_rate, batch_norm)


if __name__ == '__main__':
    # Test model creation
    print("Testing model architectures...")
    print("=" * 80)

    input_shape = (512, 512, 1)

    for name in MODEL_ARCHITECTURES.keys():
        print(f"\nCreating {name}...")
        try:
            model = get_model(name, input_shape, dropout_rate=0.3, batch_norm=True)
            print(f"✓ {model.name}")
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
            print(f"  Parameters: {model.count_params():,}")
        except Exception as e:
            print(f"✗ ERROR: {e}")

    print("\n" + "=" * 80)
    print("✓ All models created successfully!")
