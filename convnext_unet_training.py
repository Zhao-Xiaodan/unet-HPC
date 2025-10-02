#!/usr/bin/env python3
"""
ConvNeXt-UNet Dedicated Training Script

This script focuses specifically on training ConvNeXt-UNet with enhanced
cache clearing and dataset management to resolve persistent dataset conflicts.

Based on successful Swin-UNet training, optimized for ConvNeXt architecture.
"""

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
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

# Custom callback for time management and progress reporting
class TimeManagementCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_training_time_hours=22):
        super().__init__()
        self.max_training_time = max_training_time_hours * 3600  # Convert to seconds
        self.training_start_time = None
        self.epoch_times = []

    def on_train_begin(self, logs=None):
        self.training_start_time = time.time()
        print(f"🕐 Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        # Calculate statistics
        elapsed_time = time.time() - self.training_start_time
        remaining_time = self.max_training_time - elapsed_time
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)

        # Progress reporting
        if epoch % 5 == 0 or epoch < 5:  # Report every 5 epochs, or first 5 epochs
            print(f"\n📊 Progress Report (Epoch {epoch + 1}):")
            print(f"  ⏱️ Epoch time: {epoch_time:.1f}s (avg: {avg_epoch_time:.1f}s)")
            print(f"  🕐 Elapsed: {elapsed_time/3600:.1f}h, Remaining: {remaining_time/3600:.1f}h")

            if logs:
                val_jaccard = logs.get('val_jacard_coef', 0)
                val_loss = logs.get('val_loss', 0)
                print(f"  🎯 Val Jaccard: {val_jaccard:.4f}, Val Loss: {val_loss:.4f}")

        # Time management
        if remaining_time < avg_epoch_time * 2:  # Less than 2 epochs remaining
            print(f"\n⚠️ TIME WARNING: Less than 2 epochs time remaining!")
            print(f"   Stopping training to allow for cleanup and saving")
            self.model.stop_training = True

        # Emergency stop if very close to limit
        if remaining_time < 1800:  # Less than 30 minutes
            print(f"\n🚨 EMERGENCY STOP: Less than 30 minutes remaining!")
            print(f"   Forcing training stop for safe shutdown")
            self.model.stop_training = True

# =============================================================================
# Environment Setup with Enhanced Dataset Management
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

def aggressive_cache_clearing():
    """Aggressive TensorFlow cache clearing specifically for ConvNeXt-UNet dataset issues"""
    print("🧹 Performing aggressive cache clearing for ConvNeXt-UNet...")

    try:
        # Clear TensorFlow session completely
        tf.keras.backend.clear_session()

        # Reset default graph
        tf.compat.v1.reset_default_graph() if hasattr(tf.compat.v1, 'reset_default_graph') else None

        # Clear all TensorFlow cache directories aggressively
        cache_patterns = [
            os.path.expanduser('~/.tensorflow*'),
            '/tmp/tf*',
            '/tmp/tensorflow*',
            '/tmp/*tf*',
            tempfile.gettempdir() + '/tf*',
            tempfile.gettempdir() + '/*tensorflow*',
            '/var/tmp/tf*',
            '/dev/shm/tf*'
        ]

        import glob
        cleared_count = 0
        for pattern in cache_patterns:
            for path in glob.glob(pattern):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                        cleared_count += 1
                    elif os.path.isfile(path):
                        os.remove(path)
                        cleared_count += 1
                except Exception as e:
                    pass  # Ignore individual failures

        # Clear Python cache
        gc.collect()

        # Reset TensorFlow completely
        try:
            tf.compat.v1.disable_eager_execution()
            tf.compat.v1.enable_eager_execution()
        except:
            pass

        # Generate unique session ID for this run
        session_id = str(uuid.uuid4())[:8]
        os.environ['TF_SESSION_ID'] = session_id

        print(f"✓ Aggressive cache clearing completed: {cleared_count} items cleared")
        print(f"✓ Unique session ID: {session_id}")

    except Exception as e:
        print(f"Warning: Aggressive cache clearing failed: {e}")

def create_unique_dataset(X, y, session_id):
    """Create dataset with unique naming to prevent conflicts"""
    try:
        # Use tf.data.Dataset.from_tensor_slices with unique naming
        dataset_name = f"convnext_dataset_{session_id}_{int(time.time())}"

        # Convert to tensors explicitly
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

        # Create dataset without caching
        dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))

        return dataset, dataset_name

    except Exception as e:
        print(f"Warning: Unique dataset creation failed: {e}")
        return None, None

# =============================================================================
# Data Loading
# =============================================================================

def load_dataset(verbose=True):
    """Load and preprocess the mitochondria segmentation dataset."""
    if verbose:
        print("=== LOADING DATASET FOR CONVNEXT-UNET ===")

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
# ConvNeXt-UNet Training Function
# =============================================================================

def train_convnext_unet(X_train, X_test, y_train, y_test,
                       learning_rate=1e-4, batch_size=6, epochs=80, output_dir="./"):
    """
    Train ConvNeXt-UNet with enhanced dataset management.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: ConvNeXt-UNet (Dedicated)")
    print(f"Learning Rate: {learning_rate}, Batch Size: {batch_size}, Max Epochs: {epochs}")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"{'='*60}")

    # Aggressive cache clearing before starting
    aggressive_cache_clearing()

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
        # Create ConvNeXt-UNet model
        print(f"Creating ConvNeXt-UNet model...")
        model = create_modern_unet('ConvNeXt_UNet', input_shape, num_classes=1)

        # Build model explicitly
        model.build(input_shape=(None,) + input_shape)
        print(f"Model parameters: {model.count_params():,}")

        # Use AdamW optimizer for ConvNeXt with better weight decay
        try:
            from tensorflow.keras.optimizers import AdamW
            base_optimizer = AdamW(
                learning_rate=learning_rate,
                weight_decay=1e-4,  # Add weight decay for better regularization
                clipnorm=1.0
            )
            print("✓ AdamW optimizer with weight decay")
        except ImportError:
            base_optimizer = Adam(
                learning_rate=learning_rate,
                clipnorm=1.0
            )
            print("✓ Standard Adam optimizer (AdamW not available)")

        # Apply mixed precision if available
        if hasattr(tf.keras.mixed_precision, 'LossScaleOptimizer'):
            try:
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
                print("✓ Loss scaling enabled for mixed precision")
            except Exception as e:
                optimizer = base_optimizer
                print(f"⚠ Using base optimizer: {e}")
        else:
            optimizer = base_optimizer

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=BinaryFocalLoss(gamma=2),
            metrics=['accuracy', jacard_coef]
        )

        # Use Keras format instead of HDF5 to avoid dataset name conflicts
        session_id = os.environ.get('TF_SESSION_ID', 'unknown')
        timestamp_id = str(int(time.time()))[-6:]  # Last 6 digits of timestamp for uniqueness
        unique_id = f"{session_id}_{timestamp_id}"

        model_filename = f"ConvNeXt_UNet_lr{learning_rate}_bs{batch_size}_{unique_id}_model.keras"
        model_path = os.path.join(output_dir, model_filename)

        # Enhanced callbacks with Keras format to avoid HDF5 issues
        checkpoint_filepath = os.path.join(output_dir, f"ConvNeXt_UNet_checkpoint_{unique_id}.keras")

        # Use safer callback configuration to avoid HDF5 conflicts
        callbacks = [
            TimeManagementCallback(max_training_time_hours=22),  # Stop 2 hours before walltime limit
            EarlyStopping(
                monitor='val_jacard_coef',
                patience=15,  # Increased patience for better convergence
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_jacard_coef',
                factor=0.5,
                patience=8,   # Increased patience for stable learning
                min_lr=1e-7,  # Lower minimum learning rate
                mode='max',
                verbose=1
            )
        ]

        # Add ModelCheckpoint with error handling
        try:
            callbacks.append(
                ModelCheckpoint(
                    model_path,
                    monitor='val_jacard_coef',
                    save_best_only=True,
                    mode='max',
                    verbose=1,
                    save_weights_only=False  # Save full model
                )
            )
            print(f"✓ ModelCheckpoint configured: {model_path}")
        except Exception as e:
            print(f"⚠️ ModelCheckpoint configuration failed: {e}")
            print("Training will continue without automatic model saving")

        print("Starting ConvNeXt-UNet training...")
        start_time = time.time()

        # ConvNeXt-specific optimizations for speed
        print("Applying ConvNeXt-specific optimizations...")
        try:
            tf.config.experimental.enable_tensor_float_32_execution(True)  # Enable TF32 for speed
            print("✓ TF32 enabled")
        except Exception as e:
            print(f"⚠ TF32 not available: {e}")

        try:
            # Use compatible mixed precision policy
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("✓ Mixed precision (float16) enabled")
        except Exception as e:
            try:
                # Fallback to older mixed precision API
                from tensorflow.keras.mixed_precision import experimental as mixed_precision_exp
                policy = mixed_precision_exp.Policy('mixed_float16')
                mixed_precision_exp.set_policy(policy)
                print("✓ Mixed precision (experimental) enabled")
            except Exception as e2:
                print(f"⚠ Mixed precision not available: {e}, fallback failed: {e2}")

        # Enable XLA optimization if available
        try:
            tf.config.optimizer.set_jit(True)
            print("✓ XLA JIT compilation enabled")
        except Exception as e:
            print(f"⚠ XLA not available: {e}")

        # Train model with enhanced error handling
        try:
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1,
                shuffle=True,
                use_multiprocessing=False,  # Disable multiprocessing to avoid dataset conflicts
                workers=1,
                steps_per_epoch=len(X_train) // batch_size,  # Explicit steps calculation
                validation_steps=len(X_test) // batch_size   # Explicit validation steps
            )
        except Exception as training_error:
            print(f"\n⚠️ Training encountered an error: {training_error}")

            # Try to save model manually if training failed during saving
            if "Unable to synchronously create dataset" in str(training_error):
                print("🔧 Attempting manual model save with alternative format...")
                try:
                    # Use SavedModel format as fallback
                    fallback_path = os.path.join(output_dir, f"ConvNeXt_UNet_fallback_{unique_id}")
                    model.save(fallback_path, save_format='tf')
                    print(f"✓ Model saved successfully to: {fallback_path}")
                except Exception as save_error:
                    print(f"✗ Manual save also failed: {save_error}")

            # Re-raise the original error
            raise training_error

        training_time = time.time() - start_time

        # Manual model saving as backup
        print("\n💾 Saving model manually...")
        try:
            # Try Keras format first
            manual_save_path = os.path.join(output_dir, f"ConvNeXt_UNet_final_{unique_id}.keras")
            model.save(manual_save_path)
            print(f"✓ Model saved successfully: {manual_save_path}")
        except Exception as save_error1:
            print(f"⚠️ Keras format save failed: {save_error1}")
            try:
                # Fallback to SavedModel format
                manual_save_path = os.path.join(output_dir, f"ConvNeXt_UNet_final_{unique_id}_savedmodel")
                model.save(manual_save_path, save_format='tf')
                print(f"✓ Model saved with SavedModel format: {manual_save_path}")
            except Exception as save_error2:
                print(f"⚠️ SavedModel format also failed: {save_error2}")
                try:
                    # Last resort: save weights only
                    weights_path = os.path.join(output_dir, f"ConvNeXt_UNet_weights_{unique_id}.weights.h5")
                    model.save_weights(weights_path)
                    print(f"✓ Model weights saved: {weights_path}")
                except Exception as save_error3:
                    print(f"✗ All save attempts failed: {save_error3}")

        # Save training history
        history_df = pd.DataFrame(history.history)
        history_filename = f"ConvNeXt_UNet_lr{learning_rate}_bs{batch_size}_history.csv"
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
            'model_name': 'ConvNeXt_UNet',
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
            'model_parameters': int(model.count_params())
        }

        # Save results
        results_filename = f"ConvNeXt_UNet_lr{learning_rate}_bs{batch_size}_results.json"
        results_path = os.path.join(output_dir, results_filename)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ ConvNeXt-UNet training completed successfully!")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Best Val Jaccard: {best_val_jaccard:.4f} (epoch {best_epoch})")
        print(f"  Final Val Loss: {final_val_loss:.4f}")
        print(f"  Stability (std): {val_loss_stability:.4f}")

        return model, history, results

    except Exception as e:
        print(f"\n✗ ConvNeXt-UNet training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    finally:
        # Final cleanup
        aggressive_cache_clearing()

# =============================================================================
# Resume Training Function
# =============================================================================

def find_latest_checkpoint(base_dir="."):
    """Find the latest ConvNeXt-UNet checkpoint for resume training."""
    import glob

    # Look for checkpoint files (both old HDF5 and new Keras formats)
    checkpoint_patterns = [
        os.path.join(base_dir, "convnext_unet_training_*/ConvNeXt_UNet_checkpoint_*.keras"),
        os.path.join(base_dir, "ConvNeXt_UNet_checkpoint_*.keras"),
        os.path.join(base_dir, "convnext_unet_training_*/ConvNeXt_UNet_checkpoint_*.hdf5"),
        os.path.join(base_dir, "ConvNeXt_UNet_checkpoint_*.hdf5")
    ]

    all_checkpoints = []
    for pattern in checkpoint_patterns:
        all_checkpoints.extend(glob.glob(pattern))

    if not all_checkpoints:
        return None

    # Get the most recent checkpoint
    latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def check_training_completion():
    """Check if training was already completed successfully."""
    import glob

    # Look for completed training results
    result_files = glob.glob("convnext_unet_training_*/ConvNeXt_UNet_*_results.json")

    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)

            # Check if training completed with good performance
            best_jaccard = results.get('best_val_jaccard', 0)
            epochs_completed = results.get('epochs_completed', 0)

            if best_jaccard > 0.92 and epochs_completed >= 30:  # Stricter completion criteria for better performance
                print(f"✓ Found completed training with {best_jaccard:.4f} Jaccard in {epochs_completed} epochs")
                print(f"  Result file: {result_file}")
                return True, results

        except Exception as e:
            continue

    return False, None

# =============================================================================
# Main Training Script
# =============================================================================

def main():
    """Main training function for ConvNeXt-UNet."""
    print("=" * 70)
    print("CONVNEXT-UNET DEDICATED TRAINING FOR MITOCHONDRIA SEGMENTATION")
    print("=" * 70)
    print("Model: ConvNeXt-UNet (Modern CNN with improved efficiency)")
    print("Task: Mitochondria semantic segmentation")
    print("Framework: TensorFlow/Keras with enhanced dataset management")
    print()

    # Setup
    setup_gpu()

    # Check if training was already completed successfully
    print("\n🔍 Checking for previous completed training...")
    is_completed, previous_results = check_training_completion()

    if is_completed:
        print("✅ ConvNeXt-UNet training was already completed successfully!")
        print("No need to retrain. Exiting.")
        return

    # Check for checkpoints to resume from
    print("\n🔍 Checking for previous checkpoints to resume from...")
    latest_checkpoint = find_latest_checkpoint()

    if latest_checkpoint:
        print(f"✅ Found checkpoint: {latest_checkpoint}")
        print("⚠️ Resume training not implemented yet - starting fresh training")
        print("  (Future enhancement: implement resume from checkpoint)")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"convnext_unet_training_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")

    try:
        # Load dataset
        image_dataset, mask_dataset = load_dataset()
        X_train, X_test, y_train, y_test = create_data_splits(image_dataset, mask_dataset)

        # Training configuration - optimized for ConvNeXt performance
        learning_rate = 1e-4  # Reduced learning rate for better ConvNeXt convergence
        batch_size = 6        # Increased batch size for better GPU utilization
        epochs = 80           # Increased epochs for proper convergence

        print(f"\n⚙️ Training Configuration:")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Max Epochs: {epochs}")
        print(f"  Input Shape: {X_train.shape[1:]}")
        print(f"  Training Samples: {len(X_train)}")
        print(f"  Validation Samples: {len(X_test)}")
        print(f"  Steps per Epoch: {len(X_train) // batch_size}")

        # Estimate training time
        steps_per_epoch = len(X_train) // batch_size
        estimated_time_per_epoch = 20  # seconds, based on recent observation
        estimated_total_time = (steps_per_epoch * estimated_time_per_epoch * epochs) / 3600  # hours

        print(f"\n⏱️ Time Estimates:")
        print(f"  Estimated time per epoch: ~{estimated_time_per_epoch}s")
        print(f"  Estimated total training time: ~{estimated_total_time:.1f} hours")
        print(f"  Walltime limit: 24 hours")

        if estimated_total_time > 20:
            print(f"  ⚠️ Warning: Estimated time exceeds safe limit!")
            print(f"  Consider reducing epochs if training is slow")

        print(f"\n🚀 Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Train ConvNeXt-UNet
        model, history, results = train_convnext_unet(
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
            print("CONVNEXT-UNET TRAINING COMPLETED SUCCESSFULLY")
            print(f"{'='*70}")
            print(f"✓ Best Jaccard: {results['best_val_jaccard']:.4f} (epoch {results['best_epoch']})")
            print(f"✓ Training Time: {results['training_time_seconds']:.1f}s")
            print(f"✓ Parameters: {results['model_parameters']:,}")
            print(f"✓ Stability: {results['val_loss_stability']:.4f}")
            print(f"📁 All outputs saved in: {output_dir}")
        else:
            print(f"\n✗ ConvNeXt-UNet training failed!")

    except Exception as e:
        print(f"\n✗ Training script failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print("CONVNEXT-UNET DEDICATED TRAINING COMPLETED")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()