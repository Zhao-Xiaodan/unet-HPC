"""
Fixed Model Architectures Without Lambda Layers
================================================

This file contains U-Net, Attention U-Net, and Attention ResU-Net architectures
WITHOUT Lambda layers, ensuring proper Keras serialization.

Key Changes from original models.py:
1. Replaced Lambda(lambda x: K.repeat_elements()) with RepeatVector + Reshape
2. All custom layers use @keras.saving.register_keras_serializable
3. Safe for model.save() and keras.models.load_model()

References:
- Attention U-net: https://arxiv.org/pdf/1804.03999.pdf
- R2U-Net: https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
"""

import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K


# ============================================================================
# CUSTOM LAYER: RepeatElements (replaces Lambda layer)
# ============================================================================

@tf.keras.saving.register_keras_serializable(package='Custom')
class RepeatElements(layers.Layer):
    """
    Repeats elements of a tensor along an axis.

    Replaces: Lambda(lambda x: K.repeat_elements(x, rep, axis=3))

    If tensor has shape (None, 256, 256, 3), this returns shape (None, 256, 256, 6)
    when rep=2 and axis=3.
    """
    def __init__(self, rep, axis=3, **kwargs):
        super().__init__(**kwargs)
        self.rep = rep
        self.axis = axis

    def call(self, inputs):
        return K.repeat_elements(inputs, self.rep, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({
            'rep': self.rep,
            'axis': self.axis,
        })
        return config


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def repeat_elem(tensor, rep):
    """
    Repeat elements of a tensor using custom RepeatElements layer.
    No Lambda layers - fully serializable.
    """
    return RepeatElements(rep=rep, axis=3, name=f'repeat_elem_x{rep}')(tensor)


def res_conv_block(x, filter_size, size, dropout, batch_norm=False):
    """
    Residual convolutional layer.
    Two variants - activation before or after shortcut addition.

    See Fig 4 in https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
    """
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)

    # Shortcut connection
    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)

    if dropout > 0:
        res_path = layers.Dropout(dropout)(res_path)

    return res_path


def attention_block(x, gating, inter_shape):
    """
    Attention gate mechanism (FIXED: handles stride calculation properly).

    Args:
        x: Input feature map (from skip connection)
        gating: Gating signal (from decoder)
        inter_shape: Intermediate number of filters

    Returns:
        Attention-weighted feature map
    """
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    # Theta^T * x_ij + Phi^T * gating_ij + bias
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
    shape_theta_x = K.int_shape(theta_x)

    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)

    # Calculate strides safely (avoid zeros)
    stride_h = max(1, shape_theta_x[1] // shape_g[1])
    stride_w = max(1, shape_theta_x[2] // shape_g[2])

    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                       strides=(stride_h, stride_w),
                                       padding='same')(phi_g)

    # Use Add layer instead of add function for better serialization
    concat_xg = layers.Add()([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)

    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)

    # Upsample attention coefficients safely (avoid zeros)
    upsample_h = max(1, shape_x[1] // shape_theta_x[1])
    upsample_w = max(1, shape_x[2] // shape_theta_x[2])

    upsample_psi = layers.UpSampling2D(size=(upsample_h, upsample_w))(sigmoid_xg)

    # Use repeat_elem with RepeatElements layer (no Lambda!)
    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    # Attention: element-wise multiplication using Multiply layer
    y = layers.Multiply()([upsample_psi, x])

    # Final conv to match dimensions
    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)

    return result_bn


# ============================================================================
# U-NET (Standard)
# ============================================================================

def build_unet(input_shape=(512, 512, 3), n_filters=16, dropout=0.1, batch_norm=True):
    """
    Standard U-Net architecture.

    Args:
        input_shape: Input image shape (height, width, channels)
        n_filters: Number of filters in first layer (doubles each level)
        dropout: Dropout rate
        batch_norm: Whether to use batch normalization

    Returns:
        Keras model
    """
    inputs = layers.Input(input_shape, name='input')

    # Encoder (downsampling)
    c1 = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)
    if batch_norm:
        c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(c1)
    if batch_norm:
        c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    if dropout > 0:
        p1 = layers.Dropout(dropout)(p1)

    c2 = layers.Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(p1)
    if batch_norm:
        c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(c2)
    if batch_norm:
        c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    if dropout > 0:
        p2 = layers.Dropout(dropout)(p2)

    c3 = layers.Conv2D(n_filters*4, (3, 3), activation='relu', padding='same')(p2)
    if batch_norm:
        c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(n_filters*4, (3, 3), activation='relu', padding='same')(c3)
    if batch_norm:
        c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    if dropout > 0:
        p3 = layers.Dropout(dropout)(p3)

    c4 = layers.Conv2D(n_filters*8, (3, 3), activation='relu', padding='same')(p3)
    if batch_norm:
        c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(n_filters*8, (3, 3), activation='relu', padding='same')(c4)
    if batch_norm:
        c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    if dropout > 0:
        p4 = layers.Dropout(dropout)(p4)

    # Bottleneck
    c5 = layers.Conv2D(n_filters*16, (3, 3), activation='relu', padding='same')(p4)
    if batch_norm:
        c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(n_filters*16, (3, 3), activation='relu', padding='same')(c5)
    if batch_norm:
        c5 = layers.BatchNormalization()(c5)

    # Decoder (upsampling)
    u6 = layers.Conv2DTranspose(n_filters*8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    if dropout > 0:
        u6 = layers.Dropout(dropout)(u6)
    c6 = layers.Conv2D(n_filters*8, (3, 3), activation='relu', padding='same')(u6)
    if batch_norm:
        c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(n_filters*8, (3, 3), activation='relu', padding='same')(c6)
    if batch_norm:
        c6 = layers.BatchNormalization()(c6)

    u7 = layers.Conv2DTranspose(n_filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    if dropout > 0:
        u7 = layers.Dropout(dropout)(u7)
    c7 = layers.Conv2D(n_filters*4, (3, 3), activation='relu', padding='same')(u7)
    if batch_norm:
        c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(n_filters*4, (3, 3), activation='relu', padding='same')(c7)
    if batch_norm:
        c7 = layers.BatchNormalization()(c7)

    u8 = layers.Conv2DTranspose(n_filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    if dropout > 0:
        u8 = layers.Dropout(dropout)(u8)
    c8 = layers.Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(u8)
    if batch_norm:
        c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(c8)
    if batch_norm:
        c8 = layers.BatchNormalization()(c8)

    u9 = layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    if dropout > 0:
        u9 = layers.Dropout(dropout)(u9)
    c9 = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(u9)
    if batch_norm:
        c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(c9)
    if batch_norm:
        c9 = layers.BatchNormalization()(c9)

    # Output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', name='output')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs], name='UNet')
    return model


# ============================================================================
# ATTENTION U-NET
# ============================================================================

def build_attention_unet(input_shape=(512, 512, 3), n_filters=16, dropout=0.1, batch_norm=True):
    """
    Attention U-Net architecture with attention gates.

    NO Lambda layers - uses RepeatElements custom layer instead.
    Fully serializable with model.save().

    Args:
        input_shape: Input image shape
        n_filters: Number of filters in first layer
        dropout: Dropout rate
        batch_norm: Whether to use batch normalization

    Returns:
        Keras model
    """
    inputs = layers.Input(input_shape, name='input')

    # Encoder
    c1 = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)
    if batch_norm:
        c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(c1)
    if batch_norm:
        c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    if dropout > 0:
        p1 = layers.Dropout(dropout)(p1)

    c2 = layers.Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(p1)
    if batch_norm:
        c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(c2)
    if batch_norm:
        c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    if dropout > 0:
        p2 = layers.Dropout(dropout)(p2)

    c3 = layers.Conv2D(n_filters*4, (3, 3), activation='relu', padding='same')(p2)
    if batch_norm:
        c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(n_filters*4, (3, 3), activation='relu', padding='same')(c3)
    if batch_norm:
        c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    if dropout > 0:
        p3 = layers.Dropout(dropout)(p3)

    c4 = layers.Conv2D(n_filters*8, (3, 3), activation='relu', padding='same')(p3)
    if batch_norm:
        c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(n_filters*8, (3, 3), activation='relu', padding='same')(c4)
    if batch_norm:
        c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    if dropout > 0:
        p4 = layers.Dropout(dropout)(p4)

    # Bottleneck
    c5 = layers.Conv2D(n_filters*16, (3, 3), activation='relu', padding='same')(p4)
    if batch_norm:
        c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(n_filters*16, (3, 3), activation='relu', padding='same')(c5)
    if batch_norm:
        c5 = layers.BatchNormalization()(c5)

    # Decoder with attention gates
    gating = layers.Conv2DTranspose(n_filters*8, (2, 2), strides=(2, 2), padding='same')(c5)
    att4 = attention_block(c4, gating, n_filters*8)
    u6 = layers.concatenate([gating, att4])
    if dropout > 0:
        u6 = layers.Dropout(dropout)(u6)
    c6 = layers.Conv2D(n_filters*8, (3, 3), activation='relu', padding='same')(u6)
    if batch_norm:
        c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(n_filters*8, (3, 3), activation='relu', padding='same')(c6)
    if batch_norm:
        c6 = layers.BatchNormalization()(c6)

    gating = layers.Conv2DTranspose(n_filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
    att3 = attention_block(c3, gating, n_filters*4)
    u7 = layers.concatenate([gating, att3])
    if dropout > 0:
        u7 = layers.Dropout(dropout)(u7)
    c7 = layers.Conv2D(n_filters*4, (3, 3), activation='relu', padding='same')(u7)
    if batch_norm:
        c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(n_filters*4, (3, 3), activation='relu', padding='same')(c7)
    if batch_norm:
        c7 = layers.BatchNormalization()(c7)

    gating = layers.Conv2DTranspose(n_filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
    att2 = attention_block(c2, gating, n_filters*2)
    u8 = layers.concatenate([gating, att2])
    if dropout > 0:
        u8 = layers.Dropout(dropout)(u8)
    c8 = layers.Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(u8)
    if batch_norm:
        c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(c8)
    if batch_norm:
        c8 = layers.BatchNormalization()(c8)

    gating = layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    att1 = attention_block(c1, gating, n_filters)
    u9 = layers.concatenate([gating, att1])
    if dropout > 0:
        u9 = layers.Dropout(dropout)(u9)
    c9 = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(u9)
    if batch_norm:
        c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(c9)
    if batch_norm:
        c9 = layers.BatchNormalization()(c9)

    # Output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', name='output')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs], name='Attention_UNet')
    return model


# ============================================================================
# ATTENTION RESIDUAL U-NET
# ============================================================================

def build_attention_resunet(input_shape=(512, 512, 3), n_filters=16, dropout=0.1, batch_norm=True):
    """
    Attention Residual U-Net combining residual blocks with attention gates.

    NO Lambda layers - fully serializable.

    Args:
        input_shape: Input image shape
        n_filters: Number of filters in first layer
        dropout: Dropout rate
        batch_norm: Whether to use batch normalization

    Returns:
        Keras model
    """
    inputs = layers.Input(input_shape, name='input')

    # Encoder with residual blocks
    c1 = res_conv_block(inputs, 3, n_filters, dropout, batch_norm)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = res_conv_block(p1, 3, n_filters*2, dropout, batch_norm)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = res_conv_block(p2, 3, n_filters*4, dropout, batch_norm)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = res_conv_block(p3, 3, n_filters*8, dropout, batch_norm)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = res_conv_block(p4, 3, n_filters*16, dropout, batch_norm)

    # Decoder with attention gates and residual blocks
    gating = layers.Conv2DTranspose(n_filters*8, (2, 2), strides=(2, 2), padding='same')(c5)
    att4 = attention_block(c4, gating, n_filters*8)
    u6 = layers.concatenate([gating, att4])
    c6 = res_conv_block(u6, 3, n_filters*8, dropout, batch_norm)

    gating = layers.Conv2DTranspose(n_filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
    att3 = attention_block(c3, gating, n_filters*4)
    u7 = layers.concatenate([gating, att3])
    c7 = res_conv_block(u7, 3, n_filters*4, dropout, batch_norm)

    gating = layers.Conv2DTranspose(n_filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
    att2 = attention_block(c2, gating, n_filters*2)
    u8 = layers.concatenate([gating, att2])
    c8 = res_conv_block(u8, 3, n_filters*2, dropout, batch_norm)

    gating = layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(c7)
    att1 = attention_block(c1, gating, n_filters)
    u9 = layers.concatenate([gating, att1])
    c9 = res_conv_block(u9, 3, n_filters, dropout, batch_norm)

    # Output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', name='output')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs], name='Attention_ResUNet')
    return model
