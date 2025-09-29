"""
Modern U-Net Architectures for Mitochondria Segmentation

This module implements state-of-the-art U-Net variants:
- ConvNeXt-UNet: Using ConvNeXt blocks for improved feature extraction
- Swin-UNet: Incorporating Swin Transformer blocks
- CoAtNet-UNet: Combining Convolutional and Attention mechanisms

Author: Generated for mitochondria segmentation project
Based on:
- ConvNeXt: https://arxiv.org/abs/2201.03545
- Swin Transformer: https://arxiv.org/abs/2103.14030
- CoAtNet: https://arxiv.org/abs/2106.04803
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras import backend as K
import numpy as np

# Import existing metrics from the original models
from models import jacard_coef, dice_coef, dice_coef_loss, jacard_coef_loss

# =============================================================================
# ConvNeXt Building Blocks
# =============================================================================

class ConvNeXtBlock(layers.Layer):
    """ConvNeXt block implementation"""

    def __init__(self, filters, layer_scale_init_value=1e-6, drop_path_rate=0.0, **kwargs):
        super(ConvNeXtBlock, self).__init__(**kwargs)
        self.filters = filters
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path_rate = drop_path_rate

        # Depthwise convolution
        self.dwconv = layers.DepthwiseConv2D(
            kernel_size=7, padding='same'
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6)

        # Point-wise convolutions (MLP)
        self.pwconv1 = layers.Dense(4 * filters)
        self.act = layers.Activation('gelu')
        self.pwconv2 = layers.Dense(filters)

        # Projection layer for residual connection
        self.projection = layers.Conv2D(filters, 1, padding='same')

        # Layer scale
        if layer_scale_init_value > 0:
            self.gamma = self.add_weight(
                name='gamma',
                shape=(filters,),
                initializer=tf.keras.initializers.Constant(layer_scale_init_value),
                trainable=True
            )
        else:
            self.gamma = None

    def call(self, x, training=None):
        input_x = x

        # Project input to correct dimensions if needed
        if input_x.shape[-1] != self.filters:
            input_x = self.projection(input_x)

        # Depthwise convolution
        x = self.dwconv(x)

        # Permute to (N, H, W, C) for LayerNorm
        x = self.norm(x)

        # Point-wise convolutions
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Layer scale
        if self.gamma is not None:
            x = x * self.gamma

        # Residual connection
        x = input_x + x

        return x

def convnext_downsample(x, filters, kernel_size=2):
    """ConvNeXt downsampling block"""
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Conv2D(filters, kernel_size, strides=2, padding='same')(x)
    return x

def convnext_upsample(x, skip_connection, filters):
    """ConvNeXt upsampling block with skip connection"""
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip_connection])
    return x

# =============================================================================
# Swin Transformer Building Blocks
# =============================================================================

class WindowAttention(layers.Layer):
    """Simplified Window-based multi-head self attention (W-MSA) module"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, **kwargs):
        super(WindowAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Simplified to standard multi-head attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_dim,
            dropout=attn_drop
        )
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x, mask=None, training=None):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # Reshape to sequence for attention
        x_flat = tf.reshape(x, (B, H * W, C))

        # Apply multi-head attention
        x_attn = self.attention(x_flat, x_flat, training=training)
        x_attn = self.proj_drop(x_attn, training=training)

        # Reshape back to spatial
        x = tf.reshape(x_attn, (B, H, W, C))

        return x

class SwinTransformerBlock(layers.Layer):
    """Swin Transformer Block"""

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, **kwargs):
        super(SwinTransformerBlock, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )

        # Projection layer for dimension matching
        self.projection = layers.Conv2D(dim, 1, padding='same')

        self.drop_path = layers.Dropout(drop_path) if drop_path > 0.0 else layers.Identity()
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_hidden_dim),
            layers.Activation('gelu'),
            layers.Dropout(drop),
            layers.Dense(dim),
            layers.Dropout(drop)
        ])

    def call(self, x, training=None):
        H, W = tf.shape(x)[1], tf.shape(x)[2]
        B, L, C = tf.shape(x)[0], H * W, tf.shape(x)[3]

        shortcut = x
        # Project shortcut to correct dimensions if needed
        if shortcut.shape[-1] != self.dim:
            shortcut = self.projection(shortcut)

        x = self.norm1(x)

        # Window attention
        x = self.attn(x, training=training)

        # First residual connection
        x = shortcut + self.drop_path(x, training=training)

        # MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x, training=training)
        x = shortcut + self.drop_path(x, training=training)

        return x

def swin_downsample(x, filters):
    """Swin Transformer downsampling"""
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.Conv2D(filters, 2, strides=2, padding='same')(x)
    return x

def swin_upsample(x, skip_connection, filters):
    """Swin Transformer upsampling with skip connection"""
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip_connection])
    return x

# =============================================================================
# CoAtNet Building Blocks
# =============================================================================

class CoAtNetBlock(layers.Layer):
    """CoAtNet block combining convolution and attention"""

    def __init__(self, filters, use_attention=False, num_heads=4, **kwargs):
        super(CoAtNetBlock, self).__init__(**kwargs)
        self.filters = filters
        self.use_attention = use_attention
        self.num_heads = num_heads

        if use_attention:
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.attn = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=filters // num_heads,
                dropout=0.1
            )
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        else:
            self.norm1 = layers.BatchNormalization()
            self.norm2 = layers.BatchNormalization()

        # Projection layer for residual connection
        self.projection = layers.Conv2D(filters, 1, padding='same')

        # MLP layers
        self.mlp = tf.keras.Sequential([
            layers.Dense(filters * 4),
            layers.Activation('gelu'),
            layers.Dropout(0.1),
            layers.Dense(filters),
            layers.Dropout(0.1)
        ])

        # Convolution path
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.dropout = layers.Dropout(0.1)

    def call(self, x, training=None):
        if self.use_attention:
            # Self-attention path
            shortcut = x
            # Project shortcut to correct dimensions if needed
            if shortcut.shape[-1] != self.filters:
                shortcut = self.projection(shortcut)

            x = self.norm1(x)

            # Reshape for attention
            B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
            x_flat = tf.reshape(x, (B, H * W, C))

            # Apply attention
            x_attn = self.attn(x_flat, x_flat, training=training)
            x_attn = tf.reshape(x_attn, (B, H, W, C))

            x = shortcut + self.dropout(x_attn, training=training)

            # MLP path
            shortcut = x
            x = self.norm2(x)
            x = self.mlp(x, training=training)
            x = shortcut + x
        else:
            # Convolution path
            shortcut = x
            # Project shortcut to correct dimensions if needed
            if shortcut.shape[-1] != self.filters:
                shortcut = self.projection(shortcut)

            x = self.norm1(x)
            x = tf.nn.gelu(x)
            x = self.conv1(x)

            x = self.norm2(x)
            x = tf.nn.gelu(x)
            x = self.conv2(x)
            x = self.dropout(x, training=training)

            x = shortcut + x

        return x

def coatnet_downsample(x, filters):
    """CoAtNet downsampling"""
    x = layers.Conv2D(filters, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.gelu(x)
    return x

def coatnet_upsample(x, skip_connection, filters):
    """CoAtNet upsampling with skip connection"""
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip_connection])
    return x

# =============================================================================
# Modern U-Net Architectures
# =============================================================================

def ConvNeXt_UNet(input_shape, num_classes=1):
    """
    ConvNeXt-UNet implementation

    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of output classes

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(96, 4, strides=4, padding='same')(inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Encoder
    skip_connections = []

    # Stage 1
    for _ in range(3):
        x = ConvNeXtBlock(96)(x)
    skip_connections.append(x)
    x = convnext_downsample(x, 192)

    # Stage 2
    for _ in range(3):
        x = ConvNeXtBlock(192)(x)
    skip_connections.append(x)
    x = convnext_downsample(x, 384)

    # Stage 3
    for _ in range(9):
        x = ConvNeXtBlock(384)(x)
    skip_connections.append(x)
    x = convnext_downsample(x, 768)

    # Stage 4 (Bottleneck)
    for _ in range(3):
        x = ConvNeXtBlock(768)(x)

    # Decoder
    x = convnext_upsample(x, skip_connections[2], 384)
    # Project concatenated features to expected dimensions
    x = layers.Conv2D(384, 1, padding='same')(x)
    for _ in range(3):
        x = ConvNeXtBlock(384)(x)

    x = convnext_upsample(x, skip_connections[1], 192)
    # Project concatenated features to expected dimensions
    x = layers.Conv2D(192, 1, padding='same')(x)
    for _ in range(3):
        x = ConvNeXtBlock(192)(x)

    x = convnext_upsample(x, skip_connections[0], 96)
    # Project concatenated features to expected dimensions
    x = layers.Conv2D(96, 1, padding='same')(x)
    for _ in range(3):
        x = ConvNeXtBlock(96)(x)

    # Final upsampling to original resolution
    x = layers.Conv2DTranspose(48, 4, strides=4, padding='same')(x)

    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid', name='output')(x)

    model = Model(inputs, outputs, name='ConvNeXt_UNet')
    return model

def Swin_UNet(input_shape, num_classes=1, window_size=7):
    """
    Swin-UNet implementation

    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of output classes
        window_size: Window size for Swin attention

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape)

    # Patch embedding
    x = layers.Conv2D(96, 4, strides=4, padding='same')(inputs)
    x = layers.LayerNormalization(epsilon=1e-5)(x)

    # Encoder
    skip_connections = []

    # Stage 1: 96 dim
    for i in range(2):
        x = SwinTransformerBlock(
            dim=96, num_heads=3, window_size=window_size,
            shift_size=0 if i % 2 == 0 else window_size // 2
        )(x)
    skip_connections.append(x)
    x = swin_downsample(x, 192)

    # Stage 2: 192 dim
    for i in range(2):
        x = SwinTransformerBlock(
            dim=192, num_heads=6, window_size=window_size,
            shift_size=0 if i % 2 == 0 else window_size // 2
        )(x)
    skip_connections.append(x)
    x = swin_downsample(x, 384)

    # Stage 3: 384 dim
    for i in range(6):
        x = SwinTransformerBlock(
            dim=384, num_heads=12, window_size=window_size,
            shift_size=0 if i % 2 == 0 else window_size // 2
        )(x)
    skip_connections.append(x)
    x = swin_downsample(x, 768)

    # Stage 4: 768 dim (Bottleneck)
    for i in range(2):
        x = SwinTransformerBlock(
            dim=768, num_heads=24, window_size=window_size,
            shift_size=0 if i % 2 == 0 else window_size // 2
        )(x)

    # Decoder
    x = swin_upsample(x, skip_connections[2], 384)
    # Project concatenated features to expected dimensions
    x = layers.Conv2D(384, 1, padding='same')(x)
    for i in range(2):
        x = SwinTransformerBlock(
            dim=384, num_heads=12, window_size=window_size,
            shift_size=0 if i % 2 == 0 else window_size // 2
        )(x)

    x = swin_upsample(x, skip_connections[1], 192)
    # Project concatenated features to expected dimensions
    x = layers.Conv2D(192, 1, padding='same')(x)
    for i in range(2):
        x = SwinTransformerBlock(
            dim=192, num_heads=6, window_size=window_size,
            shift_size=0 if i % 2 == 0 else window_size // 2
        )(x)

    x = swin_upsample(x, skip_connections[0], 96)
    # Project concatenated features to expected dimensions
    x = layers.Conv2D(96, 1, padding='same')(x)
    for i in range(2):
        x = SwinTransformerBlock(
            dim=96, num_heads=3, window_size=window_size,
            shift_size=0 if i % 2 == 0 else window_size // 2
        )(x)

    # Final upsampling
    x = layers.Conv2DTranspose(48, 4, strides=4, padding='same')(x)

    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid', name='output')(x)

    model = Model(inputs, outputs, name='Swin_UNet')
    return model

def CoAtNet_UNet(input_shape, num_classes=1):
    """
    CoAtNet-UNet implementation combining convolution and attention

    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of output classes

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.nn.gelu(x)

    # Encoder
    skip_connections = []

    # Stage 1: Convolution blocks (64 -> 128)
    for _ in range(2):
        x = CoAtNetBlock(64, use_attention=False)(x)
    skip_connections.append(x)
    x = coatnet_downsample(x, 128)

    # Stage 2: Mixed conv + attention (128 -> 256)
    for i in range(3):
        use_attn = i >= 1  # Use attention in later blocks
        x = CoAtNetBlock(128, use_attention=use_attn, num_heads=4)(x)
    skip_connections.append(x)
    x = coatnet_downsample(x, 256)

    # Stage 3: Attention blocks (256 -> 512)
    for _ in range(4):
        x = CoAtNetBlock(256, use_attention=True, num_heads=8)(x)
    skip_connections.append(x)
    x = coatnet_downsample(x, 512)

    # Stage 4: Pure attention (Bottleneck)
    for _ in range(2):
        x = CoAtNetBlock(512, use_attention=True, num_heads=16)(x)

    # Decoder
    x = coatnet_upsample(x, skip_connections[2], 256)
    # Project concatenated features to expected dimensions
    x = layers.Conv2D(256, 1, padding='same')(x)
    for _ in range(2):
        x = CoAtNetBlock(256, use_attention=True, num_heads=8)(x)

    x = coatnet_upsample(x, skip_connections[1], 128)
    # Project concatenated features to expected dimensions
    x = layers.Conv2D(128, 1, padding='same')(x)
    for i in range(2):
        use_attn = i == 0  # Use attention in first block only
        x = CoAtNetBlock(128, use_attention=use_attn, num_heads=4)(x)

    x = coatnet_upsample(x, skip_connections[0], 64)
    # Project concatenated features to expected dimensions
    x = layers.Conv2D(64, 1, padding='same')(x)
    for _ in range(2):
        x = CoAtNetBlock(64, use_attention=False)(x)

    # Final upsampling
    x = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.gelu(x)

    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid', name='output')(x)

    model = Model(inputs, outputs, name='CoAtNet_UNet')
    return model

# =============================================================================
# Model Factory Function
# =============================================================================

def create_modern_unet(model_name, input_shape=(256, 256, 3), num_classes=1):
    """
    Factory function to create modern U-Net architectures

    Args:
        model_name: One of ['ConvNeXt_UNet', 'Swin_UNet', 'CoAtNet_UNet']
        input_shape: Input image shape
        num_classes: Number of output classes

    Returns:
        Keras Model
    """
    model_dict = {
        'ConvNeXt_UNet': ConvNeXt_UNet,
        'Swin_UNet': Swin_UNet,
        'CoAtNet_UNet': CoAtNet_UNet
    }

    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_dict.keys())}")

    return model_dict[model_name](input_shape, num_classes)

# =============================================================================
# Model Summary and Testing
# =============================================================================

if __name__ == "__main__":
    # Test model creation
    input_shape = (256, 256, 3)

    print("Creating modern U-Net models...")

    try:
        # Test ConvNeXt-UNet
        convnext_model = ConvNeXt_UNet(input_shape)
        print(f"✓ ConvNeXt-UNet created: {convnext_model.count_params():,} parameters")

        # Test Swin-UNet
        swin_model = Swin_UNet(input_shape)
        print(f"✓ Swin-UNet created: {swin_model.count_params():,} parameters")

        # Test CoAtNet-UNet
        coatnet_model = CoAtNet_UNet(input_shape)
        print(f"✓ CoAtNet-UNet created: {coatnet_model.count_params():,} parameters")

        print("\n✓ All modern U-Net models created successfully!")

    except Exception as e:
        print(f"✗ Error creating models: {e}")