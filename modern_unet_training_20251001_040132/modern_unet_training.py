#!/usr/bin/env python3
"""
Modern U-Net Architectures Training Script for Mitochondria Segmentation

This script trains state-of-the-art U-Net variants:
- ConvNeXt-UNet: Using ConvNeXt blocks for improved feature extraction
- Swin-UNet: Incorporating Swin Transformer blocks
- CoAtNet-UNet: Combining Convolutional and Attention mechanisms

Author: Generated for mitochondria segmentation project
Based on existing training framework with modern architectures
"""

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from datetime import datetime
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import json
import time

# =============================================================================
# Environment Setup
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

def install_dependencies():
    """Install required dependencies if not available."""
    import subprocess

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
        except subprocess.CalledProcessError:
            print("Creating custom focal_loss implementation...")
            create_custom_focal_loss()
            return True

def create_custom_focal_loss():
    """Create custom focal loss implementation if package not available."""
    with open('custom_focal_loss.py', 'w') as f:
        f.write('''
import tensorflow as tf
from tensorflow.keras import backend as K

class BinaryFocalLoss:
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        """Binary focal loss implementation"""
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Calculate focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)

        focal_loss = -alpha_t * K.pow(1 - pt, self.gamma) * K.log(pt)
        return K.mean(focal_loss)
''')

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_dataset(verbose=True):
    """Load and preprocess the mitochondria segmentation dataset."""
    if verbose:
        print("=== LOADING DATASET ===")

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
        raise FileNotFoundError(
            "Dataset not found! Please ensure one of the following directory structures exists:\n"
            "- dataset_full_stack/images/ and dataset_full_stack/masks/\n"
            "- dataset/images/ and dataset/masks/\n"
            "- data/images/ and data/masks/"
        )

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
        print(f"Image shape: {image_dataset[0].shape}")
        print(f"Mask shape: {mask_dataset[0].shape}")

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
# Model Training Functions
# =============================================================================

def clear_tensorflow_caches():
    """Clear TensorFlow dataset caches to prevent 'name already exists' errors"""
    import gc
    import shutil
    import tempfile

    try:
        # Clear TensorFlow session first
        tf.keras.backend.clear_session()

        # Clear various TensorFlow cache directories
        cache_dirs = [
            os.path.expanduser('~/.tensorflow_datasets'),
            '/tmp/tf_data_cache',
            '/tmp/tensorflow_cache',
            tempfile.gettempdir() + '/tf_data',
            '/tmp/tfds'
        ]

        cleared_count = 0
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    cleared_count += 1
                except Exception as e:
                    print(f"  Warning: Could not clear {cache_dir}: {e}")

        # Reset TensorFlow state
        tf.random.set_seed(None)
        gc.collect()

        if cleared_count > 0:
            print(f"  ✓ Cleared {cleared_count} TensorFlow cache directories")
        else:
            print(f"  ✓ No cache directories found to clear")

    except Exception as e:
        print(f"  Warning: Cache clearing failed: {e}")

def train_modern_unet(model_name, X_train, X_test, y_train, y_test,
                     learning_rate=1e-4, batch_size=8, epochs=100, output_dir="./"):
    """
    Train a modern U-Net architecture.

    Args:
        model_name: One of ['ConvNeXt_UNet', 'Swin_UNet', 'CoAtNet_UNet']
        X_train, X_test, y_train, y_test: Training and validation data
        learning_rate: Learning rate for optimizer
        batch_size: Batch size
        epochs: Maximum number of epochs
        output_dir: Directory to save outputs

    Returns:
        Trained model and training history
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"Learning Rate: {learning_rate}, Batch Size: {batch_size}, Max Epochs: {epochs}")
    print(f"{'='*60}")

    # Clear TensorFlow caches before each model to prevent dataset naming conflicts
    print("Clearing TensorFlow caches...")
    clear_tensorflow_caches()

    # Import models and metrics
    from modern_unet_models import create_modern_unet

    # Import existing metrics (ensure models.py exists)
    try:
        from models import jacard_coef
    except ImportError:
        # Fallback to creating custom metric
        print("⚠ models.py not found, using custom jacard_coef implementation")
        def jacard_coef(y_true, y_pred):
            import tensorflow as tf
            from tensorflow.keras import backend as K

            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)

            # Convert probabilities to binary predictions at 0.5 threshold
            y_pred_binary = K.cast(K.greater(y_pred_f, 0.5), K.floatx())

            # Calculate intersection with binary masks
            intersection = K.sum(y_true_f * y_pred_binary)
            union = K.sum(y_true_f) + K.sum(y_pred_binary) - intersection

            # Add small epsilon to prevent division by zero
            return (intersection + 1e-7) / (union + 1e-7)

    # Handle focal loss import
    try:
        from focal_loss import BinaryFocalLoss
    except ImportError:
        exec(open('custom_focal_loss.py').read())
        from custom_focal_loss import BinaryFocalLoss

    # Model configuration
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = X_train.shape[1], X_train.shape[2], X_train.shape[3]
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    # Create model
    print(f"Creating {model_name} model...")
    model = create_modern_unet(model_name, input_shape, num_classes=1)

    # Enhanced model building to ensure weights are created (critical for CoAtNet-UNet)
    try:
        # Force model building with proper input shape
        model.build(input_shape=(None,) + input_shape)
        print(f"✓ Model built successfully")

        # For complex models like CoAtNet, ensure all sublayers are built
        if 'CoAtNet' in model_name:
            print("Performing enhanced initialization for CoAtNet-UNet...")

            # Build all sublayers that might not be initialized
            for layer in model.layers:
                if hasattr(layer, 'build') and not getattr(layer, 'built', False):
                    try:
                        if hasattr(layer, 'input_spec') and layer.input_spec is not None:
                            layer.build(layer.input_spec)
                    except Exception as e:
                        print(f"Warning: Could not build layer {layer.name}: {e}")

            # Force a forward pass to initialize all weights
            dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32)
            try:
                _ = model(dummy_input, training=False)
                print("✓ Forward pass successful - all weights initialized")
            except Exception as e:
                print(f"Warning: Forward pass failed: {e}")

        print(f"Model parameters: {model.count_params():,}")

    except Exception as e:
        print(f"Warning: Could not build model explicitly: {e}")
        print(f"Model will be built on first forward pass")

        # For CoAtNet, try alternative initialization
        if 'CoAtNet' in model_name:
            print("Attempting alternative CoAtNet initialization...")
            try:
                # Create dummy input and attempt forward pass
                dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32)
                _ = model(dummy_input, training=False)
                print("✓ Alternative initialization successful")
                print(f"Model parameters: {model.count_params():,}")
            except Exception as e2:
                print(f"Alternative initialization also failed: {e2}")
                print("Model will be initialized during compilation")

    # Choose optimizer based on model type
    if 'Swin' in model_name or 'CoAtNet' in model_name:
        # Use AdamW for transformer-based models
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=0.05)
    else:
        # Use Adam for ConvNeXt
        optimizer = Adam(learning_rate=learning_rate)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=BinaryFocalLoss(gamma=2),
        metrics=['accuracy', jacard_coef]
    )

    # Callbacks
    model_filename = f"{model_name}_lr{learning_rate}_bs{batch_size}_model.hdf5"
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

    print("Starting training...")
    start_time = time.time()

    try:
        # Model-specific training optimizations
        if 'ConvNeXt' in model_name:
            print("Applying ConvNeXt-specific optimizations...")
            # Ensure no dataset caching conflicts
            tf.config.experimental.enable_tensor_float_32_execution(False)

        elif 'CoAtNet' in model_name:
            print("Applying CoAtNet-specific optimizations...")
            # Additional weight verification before training
            try:
                # Verify model can handle training step
                dummy_batch_x = X_train[:1]
                dummy_batch_y = y_train[:1]
                with tf.GradientTape() as tape:
                    predictions = model(dummy_batch_x, training=True)
                    loss = model.compiled_loss(dummy_batch_y, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                print("✓ CoAtNet gradient computation verified")
            except Exception as e:
                print(f"⚠ CoAtNet gradient verification failed: {e}")
                print("Attempting to fix weight initialization...")

                # Re-initialize model if needed
                model.compile(
                    optimizer=optimizer,
                    loss=BinaryFocalLoss(gamma=2),
                    metrics=['accuracy', jacard_coef],
                    run_eagerly=True  # Enable eager execution for debugging
                )

        # Train model
        print("Starting model training...")
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        training_time = time.time() - start_time

        # Save training history
        history_df = pd.DataFrame(history.history)
        history_filename = f"{model_name}_lr{learning_rate}_bs{batch_size}_history.csv"
        history_path = os.path.join(output_dir, history_filename)
        history_df.to_csv(history_path)

        # Calculate training metrics
        best_val_jaccard = max(history.history['val_jacard_coef'])
        best_epoch = np.argmax(history.history['val_jacard_coef']) + 1
        final_val_loss = history.history['val_loss'][-1]
        final_train_loss = history.history['loss'][-1]

        # Calculate stability metrics
        last_10_epochs = min(10, len(history.history['val_loss']))
        val_loss_stability = np.std(history.history['val_loss'][-last_10_epochs:])

        # Training results (convert numpy types to native Python types for JSON serialization)
        results = {
            'model_name': model_name,
            'learning_rate': float(learning_rate),
            'batch_size': int(batch_size),
            'epochs_completed': int(len(history.history['loss'])),
            'training_time_seconds': float(training_time),
            'best_val_jaccard': float(best_val_jaccard),
            'best_epoch': int(best_epoch),
            'final_val_loss': float(final_val_loss),
            'final_train_loss': float(final_train_loss),
            'val_loss_stability': float(val_loss_stability),
            'overfitting_gap': float(final_val_loss - final_train_loss),
            'model_parameters': int(model.count_params())
        }

        # Save results
        results_filename = f"{model_name}_lr{learning_rate}_bs{batch_size}_results.json"
        results_path = os.path.join(output_dir, results_filename)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Training completed successfully!")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Best Val Jaccard: {best_val_jaccard:.4f} (epoch {best_epoch})")
        print(f"  Final Val Loss: {final_val_loss:.4f}")
        print(f"  Stability (std): {val_loss_stability:.4f}")
        print(f"  Model saved: {model_path}")
        print(f"  History saved: {history_path}")
        print(f"  Results saved: {results_path}")

        return model, history, results

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return None, None, None

    finally:
        # Enhanced cleanup for dataset cache clearing
        try:
            import gc
            import shutil
            import tempfile

            # Clear TensorFlow session and memory
            tf.keras.backend.clear_session()
            gc.collect()

            # Clear TensorFlow dataset caches that cause "name already exists" errors
            cache_dirs = [
                os.path.expanduser('~/.tensorflow_datasets'),
                '/tmp/tf_data_cache',
                '/tmp/tensorflow_cache',
                tempfile.gettempdir() + '/tf_data'
            ]

            for cache_dir in cache_dirs:
                if os.path.exists(cache_dir):
                    try:
                        shutil.rmtree(cache_dir, ignore_errors=True)
                        print(f"✓ Cleared cache directory: {cache_dir}")
                    except Exception as e:
                        print(f"Warning: Could not clear {cache_dir}: {e}")

            # Reset TensorFlow random state to avoid naming conflicts
            tf.random.set_seed(None)

            # Force garbage collection
            gc.collect()

        except Exception as e:
            print(f"Warning: Enhanced cleanup failed: {e}")
            pass

# =============================================================================
# Visualization Functions
# =============================================================================

def plot_training_results(histories, output_dir="./"):
    """Plot training results for all models."""
    if not histories:
        print("No training histories to plot.")
        return

    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Modern U-Net Architectures Training Comparison', fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (model_name, history) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        epochs = range(1, len(history.history['loss']) + 1)

        # Training Loss
        axes[0, 0].plot(epochs, history.history['loss'], color=color, label=f'{model_name}', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Validation Loss
        axes[0, 1].plot(epochs, history.history['val_loss'], color=color, label=f'{model_name}', linewidth=2)
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Training Jaccard
        axes[1, 0].plot(epochs, history.history['jacard_coef'], color=color, label=f'{model_name}', linewidth=2)
        axes[1, 0].set_title('Training Jaccard Coefficient')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Jaccard Coefficient')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Validation Jaccard
        axes[1, 1].plot(epochs, history.history['val_jacard_coef'], color=color, label=f'{model_name}', linewidth=2)
        axes[1, 1].set_title('Validation Jaccard Coefficient')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Jaccard Coefficient')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'modern_unet_training_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training comparison plot saved: {plot_path}")

def create_performance_summary(results_list, output_dir="./"):
    """Create performance summary plot and table."""
    if not results_list:
        print("No results to summarize.")
        return

    # Create summary DataFrame
    df = pd.DataFrame(results_list)

    # Save summary CSV
    summary_path = os.path.join(output_dir, 'modern_unet_performance_summary.csv')
    df.to_csv(summary_path, index=False)
    print(f"✓ Performance summary saved: {summary_path}")

    # Create performance plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Modern U-Net Performance Summary', fontsize=14, fontweight='bold')

    models = df['model_name'].tolist()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Best Validation Jaccard
    ax = axes[0]
    jaccard_scores = df['best_val_jaccard'].tolist()
    bars = ax.bar(models, jaccard_scores, color=colors)
    ax.set_title('Best Validation Jaccard')
    ax.set_ylabel('Jaccard Coefficient')
    ax.tick_params(axis='x', rotation=45)

    for bar, score in zip(bars, jaccard_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # Training Time
    ax = axes[1]
    times = df['training_time_seconds'].tolist()
    bars = ax.bar(models, times, color=colors)
    ax.set_title('Training Time')
    ax.set_ylabel('Time (seconds)')
    ax.tick_params(axis='x', rotation=45)

    for bar, time_val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times) * 0.02,
                f'{time_val:.0f}s', ha='center', va='bottom', fontweight='bold')

    # Model Complexity
    ax = axes[2]
    params = df['model_parameters'].tolist()
    bars = ax.bar(models, params, color=colors)
    ax.set_title('Model Parameters')
    ax.set_ylabel('Parameters (millions)')
    ax.tick_params(axis='x', rotation=45)

    # Convert to millions for display
    param_millions = [p / 1e6 for p in params]
    for bar, param in zip(bars, param_millions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params) * 0.02,
                f'{param:.1f}M', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'modern_unet_performance_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Performance summary plot saved: {plot_path}")

# =============================================================================
# Main Training Script
# =============================================================================

def main():
    """Main training function."""
    print("=" * 70)
    print("MODERN U-NET ARCHITECTURES TRAINING FOR MITOCHONDRIA SEGMENTATION")
    print("=" * 70)
    print("Models: ConvNeXt-UNet, Swin-UNet, CoAtNet-UNet")
    print("Task: Mitochondria semantic segmentation")
    print("Framework: TensorFlow/Keras")
    print()

    # Setup
    setup_gpu()
    install_dependencies()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"modern_unet_training_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    try:
        # Load dataset
        image_dataset, mask_dataset = load_dataset()
        X_train, X_test, y_train, y_test = create_data_splits(image_dataset, mask_dataset)

        # Training configuration
        learning_rate = 1e-4  # Conservative learning rate for modern architectures
        batch_size = 4        # Smaller batch size due to model complexity
        epochs = 100

        print(f"\nTraining Configuration:")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Max Epochs: {epochs}")
        print(f"  Input Shape: {X_train.shape[1:]}")

        # Models to train
        models_to_train = ['ConvNeXt_UNet', 'Swin_UNet', 'CoAtNet_UNet']

        trained_models = {}
        training_histories = {}
        results_list = []

        # Train each model
        for model_name in models_to_train:
            print(f"\n\n{'='*70}")
            print(f"TRAINING MODEL {len(results_list) + 1}/{len(models_to_train)}: {model_name}")
            print(f"{'='*70}")

            model, history, results = train_modern_unet(
                model_name=model_name,
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
                trained_models[model_name] = model
                training_histories[model_name] = history
                results_list.append(results)
                print(f"✓ {model_name} training completed successfully!")
            else:
                print(f"✗ {model_name} training failed!")

        # Create visualizations and summaries
        if training_histories:
            print(f"\n{'='*70}")
            print("CREATING ANALYSIS AND VISUALIZATIONS")
            print(f"{'='*70}")

            plot_training_results(training_histories, output_dir)
            create_performance_summary(results_list, output_dir)

            # Print final summary
            print(f"\n{'='*70}")
            print("TRAINING SUMMARY")
            print(f"{'='*70}")

            if results_list:
                print(f"Successfully trained {len(results_list)}/{len(models_to_train)} models:")
                print()
                for results in results_list:
                    print(f"{results['model_name']}:")
                    print(f"  ✓ Best Jaccard: {results['best_val_jaccard']:.4f} (epoch {results['best_epoch']})")
                    print(f"  ✓ Training Time: {results['training_time_seconds']:.1f}s")
                    print(f"  ✓ Parameters: {results['model_parameters']:,}")
                    print(f"  ✓ Stability: {results['val_loss_stability']:.4f}")
                    print()

                # Find best model
                best_model = max(results_list, key=lambda x: x['best_val_jaccard'])
                print(f"🏆 Best performing model: {best_model['model_name']}")
                print(f"   Jaccard: {best_model['best_val_jaccard']:.4f}")
                print()

            print(f"📁 All outputs saved in: {output_dir}")
            print(f"   - Model files (.hdf5)")
            print(f"   - Training histories (.csv)")
            print(f"   - Results (.json)")
            print(f"   - Visualizations (.png)")

        else:
            print(f"\n✗ No models trained successfully!")

    except Exception as e:
        print(f"\n✗ Training script failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print("MODERN U-NET TRAINING COMPLETED")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()