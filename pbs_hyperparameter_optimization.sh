#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -N UNet_Hyperparameter_Optimization
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# U-NET HYPERPARAMETER OPTIMIZATION - COMPREHENSIVE GRID SEARCH
# =======================================================================
# Systematic comparison of hyperparameters across three U-Net architectures:
# - Standard U-Net, Attention U-Net, Attention ResU-Net
# Based on training dynamics analysis from architecture comparison study
# =======================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "======================================================================="
echo "U-NET HYPERPARAMETER OPTIMIZATION - GRID SEARCH"
echo "======================================================================="
echo "Testing Learning Rates: [1e-4, 5e-4, 1e-3, 5e-3]"
echo "Testing Batch Sizes: [8, 16, 32]"
echo "Testing Architectures: [UNet, Attention_UNet, Attention_ResUNet]"
echo "Epochs per experiment: 30 (reduced for grid search)"
echo "Total experiments: 4 LR × 3 BS × 3 Arch = 36 experiments"
echo "Estimated total time: 36-48 hours"
echo ""

# Job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $(free -h | grep Mem | awk '{print $2}'), CPUs: $(nproc)"
echo ""

# =======================================================================
# ENVIRONMENT SETUP
# =======================================================================

# TensorFlow environment variables
export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export CUDA_VISIBLE_DEVICES=0

# Load required modules
module load singularity

# Use the modern TensorFlow container
image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

if [ ! -f "$image" ]; then
    echo "ERROR: TensorFlow container not found at $image"
    exit 1
fi

echo "TensorFlow Container: $image"
echo ""

# =======================================================================
# HYPERPARAMETER GRID DEFINITION
# =======================================================================

# Define hyperparameter arrays
LEARNING_RATES=(1e-4 5e-4 1e-3 5e-3)
BATCH_SIZES=(8 16 32)
ARCHITECTURES=(UNet Attention_UNet Attention_ResUNet)
EPOCHS=30  # Reduced for grid search efficiency

# Create main results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_RESULTS_DIR="hyperparameter_optimization_${TIMESTAMP}"
mkdir -p "$MAIN_RESULTS_DIR"

echo "Main results directory: $MAIN_RESULTS_DIR"
echo ""

# =======================================================================
# CREATE HYPERPARAMETER TRAINING SCRIPT
# =======================================================================

echo "Creating hyperparameter training script..."

cat > hyperparameter_training.py << 'EOF'
#!/usr/bin/env python3
"""
Hyperparameter optimization training script for U-Net architectures.
Supports systematic grid search with configurable parameters.
"""

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from datetime import datetime
import cv2
from PIL import Image
from keras import backend as K
import argparse
import json

def setup_gpu():
    """Configure GPU memory growth to prevent allocation issues."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

def load_dataset():
    """Load and preprocess the mitochondria segmentation dataset."""
    print("Loading dataset...")

    # Dataset paths
    image_directory = 'dataset/images/'  # Updated path structure
    mask_directory = 'dataset/masks/'

    if not os.path.exists(image_directory) or not os.path.exists(mask_directory):
        # Fallback to data directory
        image_directory = 'data/images/'
        mask_directory = 'data/masks/'

    SIZE = 256
    image_dataset = []
    mask_dataset = []

    # Load images
    images = os.listdir(image_directory)
    for i, image_name in enumerate(images):
        if image_name.split('.')[-1] == 'tif':
            image = cv2.imread(image_directory + image_name, 1)
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            image_dataset.append(np.array(image))

    # Load masks
    masks = os.listdir(mask_directory)
    for i, image_name in enumerate(masks):
        if image_name.split('.')[-1] == 'tif':
            image = cv2.imread(mask_directory + image_name, 0)
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            mask_dataset.append(np.array(image))

    # Normalize
    image_dataset = np.array(image_dataset) / 255.0
    mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.0

    print(f"Loaded {len(image_dataset)} images and {len(mask_dataset)} masks")
    return image_dataset, mask_dataset

def create_data_splits(image_dataset, mask_dataset, random_state=42):
    """Create train/validation splits."""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        image_dataset, mask_dataset,
        test_size=0.10,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def train_model(architecture, learning_rate, batch_size, epochs, X_train, X_test, y_train, y_test, output_dir):
    """Train a specific model configuration."""

    print(f"\n{'='*60}")
    print(f"TRAINING: {architecture}")
    print(f"Learning Rate: {learning_rate}, Batch Size: {batch_size}, Epochs: {epochs}")
    print(f"{'='*60}")

    # Import models and metrics
    from models import UNet, Attention_UNet, Attention_ResUNet, jacard_coef

    # Install focal loss if needed
    try:
        from focal_loss import BinaryFocalLoss
        print("✓ focal_loss imported successfully")
    except ImportError:
        print("Installing focal_loss...")
        os.system("pip install focal-loss --user")
        from focal_loss import BinaryFocalLoss

    # Model configuration
    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    # Create model based on architecture
    if architecture == 'UNet':
        model = UNet(input_shape)
    elif architecture == 'Attention_UNet':
        model = Attention_UNet(input_shape)
    elif architecture == 'Attention_ResUNet':
        model = Attention_ResUNet(input_shape)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Compile model with hyperparameters
    model.compile(
        optimizer=Adam(lr=learning_rate, clipnorm=1.0),  # Added gradient clipping
        loss=BinaryFocalLoss(gamma=2),
        metrics=['accuracy', jacard_coef]
    )

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_jacard_coef',
            patience=10,  # Increased patience for stability
            restore_best_weights=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_jacard_coef',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='max'
        )
    ]

    # Training
    start_time = datetime.now()

    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        training_successful = True

    except Exception as e:
        print(f"Training failed: {e}")
        training_successful = False
        history = None

    end_time = datetime.now()
    training_time = end_time - start_time

    # Save results
    results = {
        'architecture': architecture,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs_requested': epochs,
        'epochs_completed': len(history.history['loss']) if history else 0,
        'training_time_seconds': training_time.total_seconds(),
        'training_successful': training_successful
    }

    if training_successful and history:
        # Extract best metrics
        best_val_jaccard = max(history.history['val_jacard_coef'])
        best_epoch = history.history['val_jacard_coef'].index(best_val_jaccard) + 1
        final_val_loss = history.history['val_loss'][-1]
        final_train_loss = history.history['loss'][-1]

        # Calculate stability metrics
        last_10_epochs = min(10, len(history.history['val_loss']))
        val_loss_stability = np.std(history.history['val_loss'][-last_10_epochs:])

        results.update({
            'best_val_jaccard': best_val_jaccard,
            'best_epoch': best_epoch,
            'final_val_loss': final_val_loss,
            'final_train_loss': final_train_loss,
            'val_loss_stability': val_loss_stability,
            'overfitting_gap': final_val_loss - final_train_loss
        })

        # Save training history
        history_df = pd.DataFrame(history.history)
        history_file = os.path.join(output_dir, f'{architecture}_lr{learning_rate}_bs{batch_size}_history.csv')
        history_df.to_csv(history_file)

        # Save model
        model_file = os.path.join(output_dir, f'{architecture}_lr{learning_rate}_bs{batch_size}_model.hdf5')
        model.save(model_file)

        print(f"\n✓ Training completed successfully!")
        print(f"  Best Val Jaccard: {best_val_jaccard:.4f} (epoch {best_epoch})")
        print(f"  Final Val Loss: {final_val_loss:.4f}")
        print(f"  Training Time: {training_time}")
        print(f"  Stability (std): {val_loss_stability:.4f}")

    else:
        print(f"\n✗ Training failed or incomplete")

    return results

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization training')
    parser.add_argument('--architecture', required=True, choices=['UNet', 'Attention_UNet', 'Attention_ResUNet'])
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output_dir', required=True)

    args = parser.parse_args()

    # Setup
    setup_gpu()

    # Load data
    image_dataset, mask_dataset = load_dataset()
    X_train, X_test, y_train, y_test = create_data_splits(image_dataset, mask_dataset)

    print(f"Training set: {X_train.shape}, Validation set: {X_test.shape}")

    # Train model
    results = train_model(
        args.architecture, args.learning_rate, args.batch_size, args.epochs,
        X_train, X_test, y_train, y_test, args.output_dir
    )

    # Save individual results
    results_file = os.path.join(args.output_dir, f'{args.architecture}_lr{args.learning_rate}_bs{args.batch_size}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
EOF

echo "✓ Hyperparameter training script created"

# =======================================================================
# EXECUTE HYPERPARAMETER GRID SEARCH
# =======================================================================

echo ""
echo "🚀 STARTING HYPERPARAMETER GRID SEARCH"
echo "======================================"

# Initialize experiment counter
experiment_num=0
total_experiments=$((${#LEARNING_RATES[@]} * ${#BATCH_SIZES[@]} * ${#ARCHITECTURES[@]}))

echo "Total experiments to run: $total_experiments"
echo ""

# Create summary file
SUMMARY_FILE="${MAIN_RESULTS_DIR}/hyperparameter_summary.csv"
echo "architecture,learning_rate,batch_size,epochs_requested,epochs_completed,training_time_seconds,training_successful,best_val_jaccard,best_epoch,final_val_loss,final_train_loss,val_loss_stability,overfitting_gap" > "$SUMMARY_FILE"

# Grid search loops
for architecture in "${ARCHITECTURES[@]}"; do
    for learning_rate in "${LEARNING_RATES[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do

            experiment_num=$((experiment_num + 1))

            echo ""
            echo "🔬 EXPERIMENT $experiment_num/$total_experiments"
            echo "================================================"
            echo "Architecture: $architecture"
            echo "Learning Rate: $learning_rate"
            echo "Batch Size: $batch_size"
            echo "Epochs: $EPOCHS"
            echo ""

            # Create experiment directory
            exp_dir="${MAIN_RESULTS_DIR}/exp_${experiment_num}_${architecture}_lr${learning_rate}_bs${batch_size}"
            mkdir -p "$exp_dir"

            # Run experiment
            echo "Starting training..."
            start_time=$(date +%s)

            singularity exec --nv "$image" python3 hyperparameter_training.py \
                --architecture "$architecture" \
                --learning_rate "$learning_rate" \
                --batch_size "$batch_size" \
                --epochs "$EPOCHS" \
                --output_dir "$exp_dir" \
                2>&1 | tee "${exp_dir}/training_log.txt"

            end_time=$(date +%s)
            experiment_time=$((end_time - start_time))

            echo ""
            echo "Experiment $experiment_num completed in ${experiment_time}s"

            # Extract results and append to summary
            results_file="${exp_dir}/${architecture}_lr${learning_rate}_bs${batch_size}_results.json"
            if [ -f "$results_file" ]; then
                python3 -c "
import json
import sys

try:
    with open('$results_file', 'r') as f:
        data = json.load(f)

    # Create CSV row
    row = [
        data.get('architecture', ''),
        data.get('learning_rate', ''),
        data.get('batch_size', ''),
        data.get('epochs_requested', ''),
        data.get('epochs_completed', ''),
        data.get('training_time_seconds', ''),
        data.get('training_successful', ''),
        data.get('best_val_jaccard', ''),
        data.get('best_epoch', ''),
        data.get('final_val_loss', ''),
        data.get('final_train_loss', ''),
        data.get('val_loss_stability', ''),
        data.get('overfitting_gap', '')
    ]

    print(','.join([str(x) for x in row]))

except Exception as e:
    print(f'ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,False,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR')
" >> "$SUMMARY_FILE"
            else
                echo "ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,False,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$SUMMARY_FILE"
            fi

            # Progress update
            remaining=$((total_experiments - experiment_num))
            estimated_remaining_time=$((remaining * experiment_time / 60))
            echo "Progress: $experiment_num/$total_experiments completed"
            echo "Estimated remaining time: ${estimated_remaining_time} minutes"

        done
    done
done

# =======================================================================
# RESULTS ANALYSIS AND SUMMARY
# =======================================================================

echo ""
echo "📊 GENERATING RESULTS ANALYSIS"
echo "=============================="

# Create analysis script
cat > analyze_hyperparameters.py << 'EOF'
#!/usr/bin/env python3
"""
Analyze hyperparameter optimization results and generate summary reports.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import argparse

def load_results(summary_file):
    """Load and clean results data."""
    df = pd.read_csv(summary_file)

    # Clean data - remove failed experiments
    df = df[df['training_successful'] == True]

    # Convert data types
    df['learning_rate'] = df['learning_rate'].astype(float)
    df['batch_size'] = df['batch_size'].astype(int)
    df['best_val_jaccard'] = df['best_val_jaccard'].astype(float)
    df['val_loss_stability'] = df['val_loss_stability'].astype(float)

    return df

def create_performance_heatmaps(df, output_dir):
    """Create heatmaps showing performance across hyperparameters."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hyperparameter Optimization Results - Performance Heatmaps', fontsize=16, fontweight='bold')

    architectures = df['architecture'].unique()

    for i, arch in enumerate(architectures):
        arch_data = df[df['architecture'] == arch]

        # Create pivot table for jaccard performance
        pivot_jaccard = arch_data.pivot_table(
            values='best_val_jaccard',
            index='learning_rate',
            columns='batch_size',
            aggfunc='mean'
        )

        # Create pivot table for stability
        pivot_stability = arch_data.pivot_table(
            values='val_loss_stability',
            index='learning_rate',
            columns='batch_size',
            aggfunc='mean'
        )

        # Plot jaccard heatmap
        sns.heatmap(
            pivot_jaccard,
            annot=True,
            fmt='.4f',
            cmap='viridis',
            ax=axes[0, i]
        )
        axes[0, i].set_title(f'{arch} - Best Val Jaccard')
        axes[0, i].set_xlabel('Batch Size')
        axes[0, i].set_ylabel('Learning Rate')

        # Plot stability heatmap
        sns.heatmap(
            pivot_stability,
            annot=True,
            fmt='.4f',
            cmap='viridis_r',  # Reverse colormap (lower is better for stability)
            ax=axes[1, i]
        )
        axes[1, i].set_title(f'{arch} - Val Loss Stability (Lower=Better)')
        axes[1, i].set_xlabel('Batch Size')
        axes[1, i].set_ylabel('Learning Rate')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hyperparameter_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_plots(df, output_dir):
    """Create comparison plots across architectures."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Architecture Comparison Across Hyperparameters', fontsize=16, fontweight='bold')

    # Performance vs Learning Rate
    for arch in df['architecture'].unique():
        arch_data = df[df['architecture'] == arch]
        lr_performance = arch_data.groupby('learning_rate')['best_val_jaccard'].mean()
        axes[0, 0].plot(lr_performance.index, lr_performance.values, marker='o', label=arch)

    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Best Val Jaccard')
    axes[0, 0].set_title('Performance vs Learning Rate')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    # Performance vs Batch Size
    for arch in df['architecture'].unique():
        arch_data = df[df['architecture'] == arch]
        bs_performance = arch_data.groupby('batch_size')['best_val_jaccard'].mean()
        axes[0, 1].plot(bs_performance.index, bs_performance.values, marker='o', label=arch)

    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Best Val Jaccard')
    axes[0, 1].set_title('Performance vs Batch Size')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Stability vs Learning Rate
    for arch in df['architecture'].unique():
        arch_data = df[df['architecture'] == arch]
        lr_stability = arch_data.groupby('learning_rate')['val_loss_stability'].mean()
        axes[1, 0].plot(lr_stability.index, lr_stability.values, marker='o', label=arch)

    axes[1, 0].set_xlabel('Learning Rate')
    axes[1, 0].set_ylabel('Val Loss Stability (Lower=Better)')
    axes[1, 0].set_title('Stability vs Learning Rate')
    axes[1, 0].legend()
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Stability vs Batch Size
    for arch in df['architecture'].unique():
        arch_data = df[df['architecture'] == arch]
        bs_stability = arch_data.groupby('batch_size')['val_loss_stability'].mean()
        axes[1, 1].plot(bs_stability.index, bs_stability.values, marker='o', label=arch)

    axes[1, 1].set_xlabel('Batch Size')
    axes[1, 1].set_ylabel('Val Loss Stability (Lower=Better)')
    axes[1, 1].set_title('Stability vs Batch Size')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'architecture_comparisons.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(df, output_dir):
    """Generate text summary report."""

    report_lines = []
    report_lines.append("HYPERPARAMETER OPTIMIZATION SUMMARY REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"Total successful experiments: {len(df)}")
    report_lines.append(f"Architectures tested: {', '.join(df['architecture'].unique())}")
    report_lines.append(f"Learning rates tested: {', '.join([str(x) for x in sorted(df['learning_rate'].unique())])}")
    report_lines.append(f"Batch sizes tested: {', '.join([str(x) for x in sorted(df['batch_size'].unique())])}")
    report_lines.append("")

    # Best overall configuration
    best_idx = df['best_val_jaccard'].idxmax()
    best_config = df.loc[best_idx]

    report_lines.append("BEST OVERALL CONFIGURATION:")
    report_lines.append("-" * 30)
    report_lines.append(f"Architecture: {best_config['architecture']}")
    report_lines.append(f"Learning Rate: {best_config['learning_rate']}")
    report_lines.append(f"Batch Size: {best_config['batch_size']}")
    report_lines.append(f"Best Val Jaccard: {best_config['best_val_jaccard']:.4f}")
    report_lines.append(f"Stability (std dev): {best_config['val_loss_stability']:.4f}")
    report_lines.append("")

    # Best configuration per architecture
    report_lines.append("BEST CONFIGURATION PER ARCHITECTURE:")
    report_lines.append("-" * 40)

    for arch in df['architecture'].unique():
        arch_data = df[df['architecture'] == arch]
        best_arch_idx = arch_data['best_val_jaccard'].idxmax()
        best_arch_config = arch_data.loc[best_arch_idx]

        report_lines.append(f"\n{arch}:")
        report_lines.append(f"  Learning Rate: {best_arch_config['learning_rate']}")
        report_lines.append(f"  Batch Size: {best_arch_config['batch_size']}")
        report_lines.append(f"  Best Val Jaccard: {best_arch_config['best_val_jaccard']:.4f}")
        report_lines.append(f"  Stability: {best_arch_config['val_loss_stability']:.4f}")

    report_lines.append("")

    # Learning rate analysis
    report_lines.append("LEARNING RATE ANALYSIS:")
    report_lines.append("-" * 25)
    lr_analysis = df.groupby('learning_rate').agg({
        'best_val_jaccard': ['mean', 'std'],
        'val_loss_stability': ['mean', 'std']
    }).round(4)

    for lr in sorted(df['learning_rate'].unique()):
        lr_data = df[df['learning_rate'] == lr]
        report_lines.append(f"\nLearning Rate {lr}:")
        report_lines.append(f"  Avg Jaccard: {lr_data['best_val_jaccard'].mean():.4f} ± {lr_data['best_val_jaccard'].std():.4f}")
        report_lines.append(f"  Avg Stability: {lr_data['val_loss_stability'].mean():.4f} ± {lr_data['val_loss_stability'].std():.4f}")

    # Batch size analysis
    report_lines.append("")
    report_lines.append("BATCH SIZE ANALYSIS:")
    report_lines.append("-" * 22)

    for bs in sorted(df['batch_size'].unique()):
        bs_data = df[df['batch_size'] == bs]
        report_lines.append(f"\nBatch Size {bs}:")
        report_lines.append(f"  Avg Jaccard: {bs_data['best_val_jaccard'].mean():.4f} ± {bs_data['best_val_jaccard'].std():.4f}")
        report_lines.append(f"  Avg Stability: {bs_data['val_loss_stability'].mean():.4f} ± {bs_data['val_loss_stability'].std():.4f}")

    # Save report
    with open(os.path.join(output_dir, 'hyperparameter_summary_report.txt'), 'w') as f:
        f.write('\n'.join(report_lines))

    # Print to console
    print('\n'.join(report_lines))

def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter optimization results')
    parser.add_argument('--summary_file', required=True, help='Path to summary CSV file')
    parser.add_argument('--output_dir', required=True, help='Output directory for plots and reports')

    args = parser.parse_args()

    # Load results
    df = load_results(args.summary_file)

    if len(df) == 0:
        print("No successful experiments found!")
        return

    print(f"Analyzing {len(df)} successful experiments...")

    # Create visualizations
    create_performance_heatmaps(df, args.output_dir)
    create_comparison_plots(df, args.output_dir)

    # Generate summary report
    generate_summary_report(df, args.output_dir)

    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
EOF

# Run analysis
echo "Running hyperparameter analysis..."
singularity exec --nv "$image" python3 analyze_hyperparameters.py \
    --summary_file "$SUMMARY_FILE" \
    --output_dir "$MAIN_RESULTS_DIR"

# =======================================================================
# FINAL SUMMARY AND CLEANUP
# =======================================================================

echo ""
echo "📋 HYPERPARAMETER OPTIMIZATION COMPLETE"
echo "======================================="
echo "Job finished on $(date)"
echo ""
echo "📁 ALL RESULTS SAVED IN: $MAIN_RESULTS_DIR"
echo ""
echo "📊 Generated Files:"
echo "  - hyperparameter_summary.csv (raw results)"
echo "  - hyperparameter_summary_report.txt (detailed analysis)"
echo "  - hyperparameter_heatmaps.png (performance heatmaps)"
echo "  - architecture_comparisons.png (comparative analysis)"
echo ""

# Display quick summary
if [ -f "$SUMMARY_FILE" ]; then
    echo "🏆 QUICK RESULTS PREVIEW:"
    echo "========================"

    # Find best result
    singularity exec --nv "$image" python3 -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('$SUMMARY_FILE')
    df = df[df['training_successful'] == True]

    if len(df) > 0:
        best_idx = df['best_val_jaccard'].idxmax()
        best = df.loc[best_idx]

        print(f'Best Overall Result:')
        print(f'  Architecture: {best[\"architecture\"]}')
        print(f'  Learning Rate: {best[\"learning_rate\"]}')
        print(f'  Batch Size: {best[\"batch_size\"]}')
        print(f'  Best Val Jaccard: {best[\"best_val_jaccard\"]:.4f}')
        print(f'  Stability: {best[\"val_loss_stability\"]:.4f}')

        print(f'\nExperiments Summary:')
        print(f'  Total Successful: {len(df)}')
        print(f'  Failed: {len(pd.read_csv(\"$SUMMARY_FILE\")) - len(df)}')

    else:
        print('No successful experiments found.')

except Exception as e:
    print(f'Error reading results: {e}')
"
fi

echo ""
echo "🔍 Next Steps:"
echo "  1. Review hyperparameter_summary_report.txt for detailed analysis"
echo "  2. Examine heatmaps for hyperparameter trends"
echo "  3. Select best configurations for production training"
echo "  4. Consider extended training with optimal hyperparameters"
echo ""
echo "✅ Hyperparameter optimization job complete!"

# Final cleanup
cd "$MAIN_RESULTS_DIR"
ls -la

echo ""
echo "======================================================================="
echo "HYPERPARAMETER OPTIMIZATION FINISHED"
echo "Results directory: $MAIN_RESULTS_DIR"
echo "======================================================================="