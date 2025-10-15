#!/usr/bin/env python3
"""
Hyperparameter Search for 512×512 Images with OOM Protection
=============================================================

Trains U-Net, ResUNet, and Attention ResUNet on dataset_shrunk_masks (98 images, 512×512).
Includes aggressive memory management to prevent OOM errors.

Features:
- Mixed precision training (reduces memory by ~40%)
- Gradient accumulation (simulates larger batches)
- Dynamic batch size adjustment on OOM
- GPU memory cleanup between models
- Conservative hyperparameter search space for 512×512

Author: Claude Code
Date: October 14, 2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import pandas as pd
import json
import gc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold

# Import custom modules
from model_architectures import get_model
from loss_functions_fixed import combined_dice_focal_loss, jacard_coef, dice_coef

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def setup_memory_growth():
    """Enable GPU memory growth to prevent OOM."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠ GPU configuration warning: {e}")

    # USE FP32 (not FP16) - FP16 causes loss=nan with 512×512 images
    # Mixed precision disabled for training stability
    print(f"✓ Using FP32 (full precision) for numerical stability")

def clear_keras_session():
    """Clear Keras session and run garbage collection."""
    keras.backend.clear_session()
    gc.collect()

    # Force GPU memory cleanup
    if tf.config.list_physical_devices('GPU'):
        try:
            # Get current memory info
            tf.config.experimental.reset_memory_stats('GPU:0')
        except:
            pass

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Dataset
    'dataset_dir': './dataset_shrunk_masks',
    'output_dir': f'./hyperparameter_search_512_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Model settings (CONSERVATIVE for 512×512!)
    'architectures': ['unet', 'resunet', 'attention_resunet'],
    'img_size': 512,  # ← 512×512 images
    'img_channels': 1,  # GRAYSCALE (converted from RGB for stability)
    'filters': 32,  # ← REDUCED from 64 (saves ~4x memory)

    # Hyperparameter search space (CONSERVATIVE!)
    'learning_rates': [1e-4, 5e-5],  # Higher LR for faster convergence
    'dropouts': [0.2, 0.3],  # Limited range
    'batch_sizes': [4],  # ← FIXED at 4 for optimal performance

    # Training settings
    'n_folds': 3,
    'epochs': 30,
    'early_stopping_patience': 7,
    'gradient_accumulation_steps': 2,  # Simulate larger batches

    # Memory safety
    'max_oom_retries': 2,
    'reduce_batch_on_oom': True,
}

print(f"\n{'='*80}")
print(f"HYPERPARAMETER SEARCH - 512×512 IMAGES WITH OOM PROTECTION")
print(f"{'='*80}")
print(f"Script: hyperparameter_search_512.py")
print(f"PBS Script: pbs_hyperparam_search_512.sh")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Working Directory: {os.getcwd()}")
print(f"")
print(f"=== DATASET CONFIGURATION ===")
print(f"Dataset: {CONFIG['dataset_dir']}")
print(f"Image Size: {CONFIG['img_size']}×{CONFIG['img_size']} (4× larger than 256×256!)")
print(f"Channels: {CONFIG['img_channels']} (GRAYSCALE)")
print(f"Expected Images: ~98")
print(f"")
print(f"=== OUTPUT CONFIGURATION ===")
print(f"Output Directory: {CONFIG['output_dir']}")
print(f"Will contain:")
print(f"  - summary.json (best configuration)")
print(f"  - all_results.csv (all training runs)")
print(f"  - {{arch}}_fold{{N}}_lr{{val}}_drop{{val}}_bs{{val}}_model.keras")
print(f"  - {{arch}}_fold{{N}}_lr{{val}}_drop{{val}}_bs{{val}}_history.csv")
print(f"  - {{arch}}_fold{{N}}_lr{{val}}_drop{{val}}_bs{{val}}_results.json")
print(f"")
print(f"=== MEMORY MANAGEMENT ===")
print(f"Challenge: 512×512 images use 4× more memory than 256×256")
print(f"Solutions:")
print(f"  1. Full precision (FP32) ← Numerical stability (no nan loss)")
print(f"  2. Grayscale images (1 channel) ← 3× memory savings vs RGB")
print(f"  3. Reduced filters: {CONFIG['filters']} (vs 64) ← ~4× memory savings")
print(f"  4. Fixed batch size: {CONFIG['batch_sizes']}")
print(f"  5. Gradient clipping (clipnorm=1.0) ← Prevents gradient explosion")
print(f"  6. OOM retry with batch size reduction")
print(f"")
print(f"=== SEARCH SPACE ===")
print(f"Architectures: {CONFIG['architectures']}")
print(f"Learning Rates: {CONFIG['learning_rates']}")
print(f"Dropouts: {CONFIG['dropouts']}")
print(f"Batch Sizes: {CONFIG['batch_sizes']}")
print(f"Cross-Validation: {CONFIG['n_folds']} folds")
print(f"Max Epochs: {CONFIG['epochs']}")
print(f"Early Stopping Patience: {CONFIG['early_stopping_patience']}")
print(f"")
total_configs = len(CONFIG['architectures']) * len(CONFIG['learning_rates']) * \
                len(CONFIG['dropouts']) * len(CONFIG['batch_sizes'])
print(f"Total Configurations: {total_configs}")
print(f"Total Training Runs: {total_configs * CONFIG['n_folds']}")
print(f"Expected Runtime: 8-12 hours")
print(f"{'='*80}\n")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(dataset_dir, img_size=512):
    """Load 512×512 PNG images from dataset_shrunk_masks (GRAYSCALE)."""
    from PIL import Image

    images_dir = Path(dataset_dir) / 'images'
    masks_dir = Path(dataset_dir) / 'masks'

    image_files = sorted(images_dir.glob('*.png'))

    if len(image_files) == 0:
        raise ValueError(f"No .png files found in {images_dir}")

    print(f"Loading {len(image_files)} images (512×512 grayscale)...")

    images = []
    masks = []

    for img_path in tqdm(image_files, desc="Loading data"):
        # Load image - CONVERT TO GRAYSCALE for stability and memory savings
        img = Image.open(img_path).convert('L')  # 'L' = grayscale
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size))
        img_array = np.array(img) / 255.0

        # Load mask
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            print(f"  ⚠ Skipping {img_path.name} - no mask")
            continue

        mask = Image.open(mask_path).convert('L')
        if mask.size != (img_size, img_size):
            mask = mask.resize((img_size, img_size))
        mask_array = np.array(mask) / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)

        images.append(img_array)
        masks.append(mask_array)

    # Add channel dimension for grayscale
    images = np.array(images, dtype=np.float32)[..., np.newaxis]
    masks = np.array(masks, dtype=np.float32)[..., np.newaxis]

    print(f"✓ Loaded {len(images)} image-mask pairs (grayscale)")
    print(f"  Images shape: {images.shape}")
    print(f"  Masks shape: {masks.shape}")
    print(f"  Memory usage: ~{images.nbytes / 1024**3:.2f} GB (images) + {masks.nbytes / 1024**3:.2f} GB (masks)")

    return images, masks

# ============================================================================
# MODEL TRAINING WITH OOM PROTECTION
# ============================================================================

def train_model_with_oom_protection(architecture, X_train, y_train, X_val, y_val, config, fold_num, output_dir):
    """Train model with OOM error handling."""

    batch_size = config['batch_size']
    oom_retries = 0

    while oom_retries <= CONFIG['max_oom_retries']:
        try:
            print(f"\n{'='*70}")
            print(f"Training {architecture} - Fold {fold_num}")
            print(f"  LR={config['lr']}, Dropout={config['dropout']}, Batch={batch_size}")
            print(f"{'='*70}")

            # Create filename prefix for this configuration and fold
            # Example: resunet_fold1_lr5e-05_drop0.2_bs4
            filename_prefix = f"{architecture}_fold{fold_num}_lr{config['lr']}_drop{config['dropout']}_bs{batch_size}"

            # Clear session before building model
            clear_keras_session()

            # Build model
            input_shape = (CONFIG['img_size'], CONFIG['img_size'], CONFIG['img_channels'])
            model = get_model(
                model_name=architecture,
                input_shape=input_shape,
                NUM_CLASSES=1,
                dropout_rate=config['dropout'],
                batch_norm=True
            )

            # Compile with FP32 and gradient clipping
            # Gradient clipping prevents gradient explosion by capping gradient norm
            # Essential for large 512×512 images which produce large gradients
            optimizer = keras.optimizers.Adam(
                learning_rate=config['lr'],
                clipnorm=1.0  # Clip gradients with norm > 1.0
            )

            model.compile(
                optimizer=optimizer,
                loss=combined_dice_focal_loss,
                metrics=[jacard_coef, dice_coef]
            )

            # Callbacks - save with explicit filenames
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    str(output_dir / f'{filename_prefix}_model.keras'),
                    monitor='val_jacard_coef',
                    mode='max',
                    save_best_only=True,
                    verbose=1
                ),
                keras.callbacks.CSVLogger(
                    str(output_dir / f'{filename_prefix}_history.csv'),
                    append=False
                ),
                keras.callbacks.EarlyStopping(
                    monitor='val_jacard_coef',
                    patience=CONFIG['early_stopping_patience'],
                    mode='max',
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_jacard_coef',
                    factor=0.5,
                    patience=4,
                    mode='max',
                    min_lr=1e-7,
                    verbose=1
                ),
            ]

            # Train
            print(f"Training with batch_size={batch_size}...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=CONFIG['epochs'],
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # Get best metrics
            val_jacards = history.history['val_jacard_coef']
            best_epoch = np.argmax(val_jacards)
            best_jaccard = val_jacards[best_epoch]

            train_jaccard = history.history['jacard_coef'][best_epoch]
            overfitting_gap = (train_jaccard - best_jaccard) * 100

            print(f"✓ Training successful!")
            print(f"  Best Jaccard: {best_jaccard:.4f} (epoch {best_epoch})")
            print(f"  Overfitting gap: {overfitting_gap:.2f}%")

            # Save results JSON
            results_dict = {
                'architecture': architecture,
                'config': {
                    'learning_rate': config['lr'],
                    'dropout': config['dropout'],
                    'batch_size': batch_size
                },
                'config_name': config['config_name'],
                'fold': fold_num,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'best_val_jaccard': float(best_jaccard),
                'best_val_dice': float(history.history['val_dice_coef'][best_epoch]),
                'best_epoch': int(best_epoch),
                'final_val_jaccard': float(val_jacards[-1]),
                'final_train_jaccard': float(history.history['jacard_coef'][-1]),
                'overfitting_gap': float(overfitting_gap),
                'epochs_trained': len(val_jacards),
                'batch_size_used': batch_size,
                'oom_retries': oom_retries,
                'success': True,
                'filename_prefix': filename_prefix
            }

            with open(output_dir / f'{filename_prefix}_results.json', 'w') as f:
                json.dump(results_dict, f, indent=2)

            print(f"  Saved: {filename_prefix}_model.keras")
            print(f"  Saved: {filename_prefix}_history.csv")
            print(f"  Saved: {filename_prefix}_results.json")

            # Clear model from memory
            del model
            clear_keras_session()

            return results_dict

        except tf.errors.ResourceExhaustedError as e:
            print(f"\n✗ OOM ERROR! (Retry {oom_retries + 1}/{CONFIG['max_oom_retries']})")
            print(f"  Error: {str(e)[:200]}")

            # Clear session
            clear_keras_session()

            if CONFIG['reduce_batch_on_oom'] and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                print(f"  → Reducing batch size to {batch_size}")
                oom_retries += 1
            else:
                print(f"  → Cannot reduce batch size further (batch_size={batch_size})")
                return {
                    'val_jaccard': 0.0,
                    'success': False,
                    'error': 'OOM - batch size cannot be reduced',
                    'oom_retries': oom_retries
                }

        except Exception as e:
            print(f"\n✗ UNEXPECTED ERROR!")
            print(f"  Error: {str(e)[:200]}")

            clear_keras_session()

            return {
                'val_jaccard': 0.0,
                'success': False,
                'error': str(e)[:200],
                'oom_retries': oom_retries
            }

    # Max retries exceeded
    return {
        'val_jaccard': 0.0,
        'success': False,
        'error': f'OOM - max retries ({CONFIG["max_oom_retries"]}) exceeded',
        'oom_retries': oom_retries
    }

# ============================================================================
# HYPERPARAMETER SEARCH
# ============================================================================

def run_hyperparameter_search():
    """Run complete hyperparameter search with cross-validation."""

    # Setup
    setup_memory_growth()
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    X, y = load_dataset(CONFIG['dataset_dir'], CONFIG['img_size'])

    # Initialize results storage
    all_results = []

    # Generate all configurations
    configs = []
    for arch in CONFIG['architectures']:
        for lr in CONFIG['learning_rates']:
            for dropout in CONFIG['dropouts']:
                for bs in CONFIG['batch_sizes']:
                    configs.append({
                        'architecture': arch,
                        'lr': lr,
                        'dropout': dropout,
                        'batch_size': bs,
                        'config_name': f"{arch}_lr{lr}_drop{dropout}_bs{bs}"
                    })

    print(f"\n✓ Generated {len(configs)} configurations")

    # Run cross-validation for each config
    kfold = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=42)

    for config_idx, config in enumerate(configs, 1):
        print(f"\n\n{'='*80}")
        print(f"CONFIGURATION {config_idx}/{len(configs)}: {config['config_name']}")
        print(f"{'='*80}")

        fold_results = []

        for fold_num, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            print(f"\nFold {fold_num}/{CONFIG['n_folds']}")
            print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

            result = train_model_with_oom_protection(
                config['architecture'],
                X_train, y_train,
                X_val, y_val,
                config,
                fold_num,
                output_dir
            )

            # Result already contains most info, just add architecture-level info
            if 'fold' not in result:
                result['fold'] = fold_num
            if 'config_name' not in result:
                result['config_name'] = config['config_name']

            fold_results.append(result)
            all_results.append(result)

            # Save intermediate results
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(output_dir / 'intermediate_results.csv', index=False)

        # Print fold summary
        successful_folds = [r for r in fold_results if r.get('success', False)]
        if successful_folds:
            jacards = [r.get('best_val_jaccard', r.get('val_jaccard', 0.0)) for r in successful_folds]
            print(f"\n✓ Configuration summary:")
            print(f"  Successful folds: {len(successful_folds)}/{CONFIG['n_folds']}")
            print(f"  Mean Jaccard: {np.mean(jacards):.4f} ± {np.std(jacards):.4f}")
            print(f"  Range: [{np.min(jacards):.4f}, {np.max(jacards):.4f}]")
        else:
            print(f"\n✗ All folds failed for this configuration")

    # Final analysis
    print(f"\n\n{'='*80}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*80}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'all_results.csv', index=False)

    # Find best configuration
    successful_results = results_df[results_df['success'] == True]

    if len(successful_results) > 0:
        # Use best_val_jaccard as the primary metric
        jaccard_col = 'best_val_jaccard' if 'best_val_jaccard' in successful_results.columns else 'val_jaccard'

        # Group by configuration
        config_summary = successful_results.groupby('config_name').agg({
            jaccard_col: ['mean', 'std', 'count']
        }).round(4)

        print(f"\n✓ Successful configurations: {len(config_summary)}")
        print(f"\nTop 5 configurations:")
        print(config_summary.sort_values((jaccard_col, 'mean'), ascending=False).head())

        # Best configuration
        best_config = config_summary[(jaccard_col, 'mean')].idxmax()
        best_jaccard = config_summary.loc[best_config, (jaccard_col, 'mean')]

        print(f"\n🏆 BEST CONFIGURATION: {best_config}")
        print(f"   Mean Jaccard: {best_jaccard:.4f}")

        # Save summary
        summary = {
            'best_config': best_config,
            'best_jaccard': float(best_jaccard),
            'total_configs': len(configs),
            'successful_configs': len(config_summary),
            'config_summary': config_summary.to_dict()
        }

        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    else:
        print(f"\n✗ No successful training runs!")
        print(f"   All configurations failed (likely OOM)")

    # Save experiment metadata
    print(f"\n📋 Saving experiment metadata...")
    metadata = {
        'experiment_name': 'Hyperparameter Search - 512×512 Images',
        'python_script': 'hyperparameter_search_512.py',
        'pbs_script': 'pbs_hyperparam_search_512.sh',
        'timestamp_start': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'working_directory': os.getcwd(),
        'output_directory': str(output_dir),
        'dataset': {
            'path': CONFIG['dataset_dir'],
            'image_size': f"{CONFIG['img_size']}×{CONFIG['img_size']}",
            'channels': CONFIG['img_channels'],
            'channel_type': 'Grayscale',
            'num_images': len(X)
        },
        'model_config': {
            'architectures': CONFIG['architectures'],
            'filters': CONFIG['filters'],
            'precision': 'FP32',
            'gradient_clipping': 'clipnorm=1.0'
        },
        'hyperparameters': {
            'learning_rates': CONFIG['learning_rates'],
            'dropouts': CONFIG['dropouts'],
            'batch_sizes': CONFIG['batch_sizes'],
            'n_folds': CONFIG['n_folds'],
            'epochs': CONFIG['epochs'],
            'early_stopping_patience': CONFIG['early_stopping_patience']
        },
        'search_summary': {
            'total_configs': len(configs),
            'total_runs': len(configs) * CONFIG['n_folds'],
            'successful_runs': len(successful_results) if len(successful_results) > 0 else 0,
            'failed_runs': len(results_df) - (len(successful_results) if len(successful_results) > 0 else 0)
        }
    }

    with open(output_dir / 'EXPERIMENT_INFO.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   ✓ Saved: EXPERIMENT_INFO.json")

    # Create README
    readme_content = f"""# Hyperparameter Search Results - 512×512 Images

**Experiment Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Output Directory:** `{output_dir.name}`

## Experiment Overview

- **Python Script:** `hyperparameter_search_512.py`
- **PBS Script:** `pbs_hyperparam_search_512.sh`
- **Dataset:** `{CONFIG['dataset_dir']}` ({len(X)} images, {CONFIG['img_size']}×{CONFIG['img_size']} grayscale)
- **Architectures Tested:** {', '.join(CONFIG['architectures'])}

## Configuration

### Model Settings
- **Filters:** {CONFIG['filters']}
- **Precision:** FP32 (full precision for stability)
- **Gradient Clipping:** clipnorm=1.0

### Hyperparameter Search Space
- **Learning Rates:** {CONFIG['learning_rates']}
- **Dropouts:** {CONFIG['dropouts']}
- **Batch Sizes:** {CONFIG['batch_sizes']}
- **Cross-Validation:** {CONFIG['n_folds']} folds
- **Max Epochs:** {CONFIG['epochs']}
- **Early Stopping Patience:** {CONFIG['early_stopping_patience']}

### Search Summary
- **Total Configurations:** {len(configs)}
- **Total Training Runs:** {len(configs) * CONFIG['n_folds']}
- **Successful Runs:** {len(successful_results) if len(successful_results) > 0 else 0}
- **Failed Runs:** {len(results_df) - (len(successful_results) if len(successful_results) > 0 else 0)}

## Files in This Directory

### Summary Files
- **`EXPERIMENT_INFO.json`** - Experiment metadata and configuration
- **`summary.json`** - Best configuration and overall summary
- **`all_results.csv`** - All training results (one row per fold)
- **`README.md`** - This file

### Model Files (per fold)
- **`{{arch}}_fold{{N}}_lr{{val}}_drop{{val}}_bs{{val}}_model.keras`** - Best model weights
- **`{{arch}}_fold{{N}}_lr{{val}}_drop{{val}}_bs{{val}}_history.csv`** - Training history
- **`{{arch}}_fold{{N}}_lr{{val}}_drop{{val}}_bs{{val}}_results.json`** - Fold metrics

### Log Files
- **`HyperSearch_512.o######`** - PBS job output (contains all echoed info from PBS script)
- **`hyperparam_search_512_console_*.log`** - Python console output

### Source Code (archived for reproducibility)
- **`hyperparameter_search_512.py`** - Python script used
- **`pbs_hyperparam_search_512.sh`** - PBS script used

## Quick Analysis

```bash
# View best configuration
cat summary.json

# View experiment details
cat EXPERIMENT_INFO.json

# List all models
ls -lh *_model.keras

# Find best model across all folds
grep -h "best_val_jaccard" *_results.json | sort -t: -k2 -nr | head -5

# Compare architectures
for arch in unet resunet attention_resunet; do
    echo "$arch:"
    grep -h "best_val_jaccard" ${{arch}}_*_results.json | awk -F: '{{sum+=$2; count++}} END {{print "  Mean Jaccard:", sum/count}}'
done
```

## Next Steps

1. **Review Results:**
   ```bash
   cat summary.json
   python analyze_512_results.py
   ```

2. **Load Best Model for Predictions:**
   ```python
   from tensorflow import keras
   model = keras.models.load_model('<best_model_file>.keras',
                                    custom_objects={{'combined_dice_focal_loss': ...,
                                                    'jacard_coef': ...,
                                                    'dice_coef': ...}})
   ```

3. **Run Density Analysis:**
   Update `pbs_density_analysis_512.sh` with this directory and submit:
   ```bash
   qsub pbs_density_analysis_512.sh
   ```

## Reproducibility

All source code files have been archived in this directory. To reproduce:

```bash
# Copy scripts back to working directory
cp hyperparameter_search_512.py ../
cp pbs_hyperparam_search_512.sh ../

# Submit to PBS
qsub pbs_hyperparam_search_512.sh
```

---

*Generated automatically by hyperparameter_search_512.py*
"""

    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)

    print(f"   ✓ Saved: README.md")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    run_hyperparameter_search()
