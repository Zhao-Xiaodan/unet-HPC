#!/usr/bin/env python3
"""
CoAtNet-UNet Dedicated Training Script

This script focuses specifically on training CoAtNet-UNet with enhanced
weight initialization and gradient computation verification to resolve
persistent trainable_variables initialization issues.

Based on successful Swin-UNet training, optimized for CoAtNet hybrid architecture.
"""

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from datetime import datetime
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import json
import time
import gc
import shutil
import tempfile
import uuid

# =============================================================================
# Environment Setup with Enhanced Weight Initialization Management
# =============================================================================

def setup_gpu():
    """Configure GPU memory growth to prevent allocation issues."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPUs")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("No GPUs found. Running on CPU.")
        return False

def comprehensive_session_reset():
    """Comprehensive TensorFlow session reset for CoAtNet-UNet training"""
    print("🔄 Performing comprehensive session reset for CoAtNet-UNet...")

    try:
        # Clear TensorFlow session completely
        tf.keras.backend.clear_session()

        # Reset default graph
        if hasattr(tf.compat.v1, 'reset_default_graph'):
            tf.compat.v1.reset_default_graph()

        # Force garbage collection
        gc.collect()

        # Reset TensorFlow eager execution
        try:
            if tf.executing_eagerly():
                tf.compat.v1.disable_eager_execution()
                tf.compat.v1.enable_eager_execution()
        except:
            pass

        # Generate unique session ID for this run
        session_id = str(uuid.uuid4())[:8]
        os.environ['TF_COATNET_SESSION_ID'] = session_id

        print(f"✓ Comprehensive session reset completed")
        print(f"✓ Unique CoAtNet session ID: {session_id}")

        return session_id

    except Exception as e:
        print(f"Warning: Session reset failed: {e}")
        return None

def enhanced_model_initialization(model, input_shape, model_name="CoAtNet_UNet"):
    """Enhanced model initialization specifically for CoAtNet-UNet architecture"""
    print(f"🔧 Applying enhanced initialization for {model_name}...")

    try:
        # Step 1: Force model building with proper input shape
        model.build(input_shape=(None,) + input_shape)
        print(f"✓ Model built with input shape: {input_shape}")

        # Step 2: Initialize all sublayers explicitly for complex hybrid architecture
        layer_count = 0
        for layer in model.layers:
            if hasattr(layer, 'build') and not getattr(layer, 'built', False):
                try:
                    if hasattr(layer, 'input_spec') and layer.input_spec is not None:
                        layer.build(layer.input_spec)
                        layer_count += 1
                    elif hasattr(layer, 'input_shape'):
                        layer.build(layer.input_shape)
                        layer_count += 1
                except Exception as e:
                    print(f"Warning: Could not build layer {layer.name}: {e}")

        print(f"✓ Built {layer_count} additional sublayers")

        # Step 3: Force forward pass to initialize all weights
        dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32)
        try:
            _ = model(dummy_input, training=False)
            print("✓ Initial forward pass successful")
        except Exception as e:
            print(f"Warning: Initial forward pass failed: {e}")
            # Try with training=True if training=False fails
            try:
                _ = model(dummy_input, training=True)
                print("✓ Training forward pass successful")
            except Exception as e2:
                print(f"Warning: Training forward pass also failed: {e2}")

        # Step 4: Verify gradient computation
        try:
            with tf.GradientTape() as tape:
                predictions = model(dummy_input, training=True)
                dummy_target = tf.zeros_like(predictions)
                loss = tf.reduce_mean(tf.square(predictions - dummy_target))

            gradients = tape.gradient(loss, model.trainable_variables)

            # Check gradient validity
            valid_gradients = sum(1 for grad in gradients if grad is not None)
            total_variables = len(model.trainable_variables)

            print(f"✓ Gradient computation: {valid_gradients}/{total_variables} variables have valid gradients")

            if valid_gradients < total_variables:
                print(f"⚠ Warning: {total_variables - valid_gradients} variables have None gradients")
                return False

            return True

        except Exception as e:
            print(f"✗ Gradient computation failed: {e}")
            return False

    except Exception as e:
        print(f"✗ Enhanced initialization failed: {e}")
        return False

# =============================================================================
# Data Loading
# =============================================================================

def load_dataset(verbose=True):
    """Load and preprocess the mitochondria segmentation dataset."""
    if verbose:
        print("=== LOADING DATASET FOR COATNET-UNET ===")

    # Dataset paths - prioritize full stack dataset
    dataset_dirs = [
        ('dataset_full_stack/images/', 'dataset_full_stack/masks/'),
        ('dataset/images/', 'dataset/masks/'),
        ('data/images/', 'data/masks/')
    ]

    image_directory = None
    mask_directory = None

    for img_dir, mask_dir in dataset_dirs:
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            image_directory = img_dir
            mask_directory = mask_dir
            break

    if not image_directory:
        raise FileNotFoundError("Dataset not found!")

    if verbose:
        print(f"Using dataset: {image_directory} and {mask_directory}")

    SIZE = 256
    image_dataset = []
    mask_dataset = []

    # Load images
    images = os.listdir(image_directory)
    for i, image_name in enumerate(images):
        if image_name.split('.')[-1] in ['tif', 'tiff', 'png', 'jpg']:
            image = cv2.imread(image_directory + image_name, 1)
            if image is not None:
                image = Image.fromarray(image)
                image = image.resize((SIZE, SIZE))
                image_dataset.append(np.array(image))

    # Load masks
    masks = os.listdir(mask_directory)
    for i, image_name in enumerate(masks):
        if image_name.split('.')[-1] in ['tif', 'tiff', 'png', 'jpg']:
            image = cv2.imread(mask_directory + image_name, 0)
            if image is not None:
                image = Image.fromarray(image)
                image = image.resize((SIZE, SIZE))
                mask_dataset.append(np.array(image))

    if len(image_dataset) == 0 or len(mask_dataset) == 0:
        raise ValueError("No valid images found in dataset directories!")

    # Normalize
    image_dataset = np.array(image_dataset) / 255.0
    mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.0

    if verbose:
        print(f"✓ Loaded {len(image_dataset)} images and {len(mask_dataset)} masks")

    return image_dataset, mask_dataset

def create_data_splits(image_dataset, mask_dataset, test_size=0.10, random_state=42):
    """Create train/validation splits."""
    X_train, X_test, y_train, y_test = train_test_split(
        image_dataset, mask_dataset,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test

# =============================================================================
# CoAtNet-UNet Training Function
# =============================================================================

def train_coatnet_unet(X_train, X_test, y_train, y_test,
                      learning_rate=2e-4, batch_size=3, epochs=100, output_dir="./"):
    """
    Train CoAtNet-UNet with enhanced weight initialization and gradient verification.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: CoAtNet-UNet (Dedicated)")
    print(f"Learning Rate: {learning_rate}, Batch Size: {batch_size}, Max Epochs: {epochs}")
    print(f"{'='*60}")

    # Comprehensive session reset before starting
    session_id = comprehensive_session_reset()

    # Import models and metrics
    from modern_unet_models import create_modern_unet

    # Import existing metrics
    try:
        from models import jacard_coef
    except ImportError:
        def jacard_coef(y_true, y_pred):
            import tensorflow as tf
            from tensorflow.keras import backend as K

            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            y_pred_binary = K.cast(K.greater(y_pred_f, 0.5), K.floatx())
            intersection = K.sum(y_true_f * y_pred_binary)
            union = K.sum(y_true_f) + K.sum(y_pred_binary) - intersection
            return (intersection + 1e-7) / (union + 1e-7)

    # Handle focal loss
    try:
        from focal_loss import BinaryFocalLoss
    except ImportError:
        class BinaryFocalLoss:
            def __init__(self, gamma=2.0, alpha=0.25):
                self.gamma = gamma
                self.alpha = alpha

            def __call__(self, y_true, y_pred):
                from tensorflow.keras import backend as K
                y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
                pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
                alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
                focal_loss = -alpha_t * K.pow(1 - pt, self.gamma) * K.log(pt)
                return K.mean(focal_loss)

    # Model configuration
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = X_train.shape[1], X_train.shape[2], X_train.shape[3]
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    try:
        # Create CoAtNet-UNet model
        print(f"Creating CoAtNet-UNet model...")
        model = create_modern_unet('CoAtNet_UNet', input_shape, num_classes=1)

        # Apply enhanced initialization for CoAtNet
        initialization_success = enhanced_model_initialization(model, input_shape, "CoAtNet_UNet")

        if not initialization_success:
            print("⚠ Warning: Enhanced initialization had issues, but proceeding with training...")

        print(f"Model parameters: {model.count_params():,}")

        # Use AdamW optimizer for CoAtNet (better for hybrid architectures)
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-4, clipnorm=1.0)

        # Compile model with eager execution for better debugging
        model.compile(
            optimizer=optimizer,
            loss=BinaryFocalLoss(gamma=2),
            metrics=['accuracy', jacard_coef],
            run_eagerly=False  # Start with False, enable if gradient issues
        )

        # Test compilation with small batch
        try:
            dummy_batch_x = X_train[:2]
            dummy_batch_y = y_train[:2]

            loss_value = model.evaluate(dummy_batch_x, dummy_batch_y, verbose=0)
            print("✓ Model compilation test successful")

        except Exception as e:
            print(f"⚠ Model compilation test failed: {e}")
            print("Enabling eager execution for better error handling...")
            model.compile(
                optimizer=optimizer,
                loss=BinaryFocalLoss(gamma=2),
                metrics=['accuracy', jacard_coef],
                run_eagerly=True
            )

        # Callbacks
        model_filename = f"CoAtNet_UNet_lr{learning_rate}_bs{batch_size}_model.hdf5"
        model_path = os.path.join(output_dir, model_filename)

        callbacks = [
            EarlyStopping(
                monitor='val_jacard_coef',
                patience=15,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_jacard_coef',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_jacard_coef',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                mode='max',
                verbose=1
            )
        ]

        print("Starting CoAtNet-UNet training...")
        start_time = time.time()

        # CoAtNet-specific optimizations
        print("Applying CoAtNet-specific optimizations...")
        tf.config.experimental.enable_tensor_float_32_execution(True)  # Enable for hybrid architectures

        # Train model with careful batch handling for CoAtNet
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
            use_multiprocessing=False,  # Disable multiprocessing for stability
            workers=1
        )

        training_time = time.time() - start_time

        # Save training history
        history_df = pd.DataFrame(history.history)
        history_filename = f"CoAtNet_UNet_lr{learning_rate}_bs{batch_size}_history.csv"
        history_path = os.path.join(output_dir, history_filename)
        history_df.to_csv(history_path)

        # Calculate metrics
        best_val_jaccard = float(max(history.history['val_jacard_coef']))
        best_epoch = int(np.argmax(history.history['val_jacard_coef']) + 1)
        final_val_loss = float(history.history['val_loss'][-1])
        final_train_loss = float(history.history['loss'][-1])

        # Calculate stability
        last_10_epochs = min(10, len(history.history['val_loss']))
        val_loss_stability = float(np.std(history.history['val_loss'][-last_10_epochs:]))

        # Training results
        results = {
            'model_name': 'CoAtNet_UNet',
            'learning_rate': float(learning_rate),
            'batch_size': int(batch_size),
            'epochs_completed': int(len(history.history['loss'])),
            'training_time_seconds': float(training_time),
            'best_val_jaccard': best_val_jaccard,
            'best_epoch': best_epoch,
            'final_val_loss': final_val_loss,
            'final_train_loss': final_train_loss,
            'val_loss_stability': val_loss_stability,
            'overfitting_gap': float(final_val_loss - final_train_loss),
            'model_parameters': int(model.count_params()),
            'initialization_success': initialization_success
        }

        # Save results
        results_filename = f"CoAtNet_UNet_lr{learning_rate}_bs{batch_size}_results.json"
        results_path = os.path.join(output_dir, results_filename)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ CoAtNet-UNet training completed successfully!")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Best Val Jaccard: {best_val_jaccard:.4f} (epoch {best_epoch})")
        print(f"  Final Val Loss: {final_val_loss:.4f}")
        print(f"  Stability (std): {val_loss_stability:.4f}")

        return model, history, results

    except Exception as e:
        print(f"\n✗ CoAtNet-UNet training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    finally:
        # Final cleanup
        comprehensive_session_reset()

# =============================================================================
# Main Training Script
# =============================================================================

def main():
    """Main training function for CoAtNet-UNet."""
    print("=" * 70)
    print("COATNET-UNET DEDICATED TRAINING FOR MITOCHONDRIA SEGMENTATION")
    print("=" * 70)
    print("Model: CoAtNet-UNet (Hybrid Attention + Convolution)")
    print("Task: Mitochondria semantic segmentation")
    print("Framework: TensorFlow/Keras with enhanced weight initialization")
    print()

    # Setup
    setup_gpu()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"coatnet_unet_training_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    try:
        # Load dataset
        image_dataset, mask_dataset = load_dataset()
        X_train, X_test, y_train, y_test = create_data_splits(image_dataset, mask_dataset)

        # Training configuration optimized for CoAtNet
        learning_rate = 2e-4  # Higher learning rate for hybrid architectures
        batch_size = 3       # Smaller batch size due to higher memory requirements
        epochs = 100

        print(f"\nTraining Configuration:")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Max Epochs: {epochs}")
        print(f"  Input Shape: {X_train.shape[1:]}")

        # Train CoAtNet-UNet
        model, history, results = train_coatnet_unet(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            output_dir=output_dir
        )

        if model is not None and results is not None:
            print(f"\n{'='*70}")
            print("COATNET-UNET TRAINING COMPLETED SUCCESSFULLY")
            print(f"{'='*70}")
            print(f"✓ Best Jaccard: {results['best_val_jaccard']:.4f} (epoch {results['best_epoch']})")
            print(f"✓ Training Time: {results['training_time_seconds']:.1f}s")
            print(f"✓ Parameters: {results['model_parameters']:,}")
            print(f"✓ Stability: {results['val_loss_stability']:.4f}")
            print(f"✓ Initialization Success: {results['initialization_success']}")
            print(f"📁 All outputs saved in: {output_dir}")
        else:
            print(f"\n✗ CoAtNet-UNet training failed!")

    except Exception as e:
        print(f"\n✗ Training script failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print("COATNET-UNET DEDICATED TRAINING COMPLETED")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()