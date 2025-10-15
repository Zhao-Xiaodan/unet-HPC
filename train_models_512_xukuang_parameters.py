================================================================================
TRAINING SCRIPT FOR U-NET, ATTENTION U-NET, AND ATTENTION RES-UNET
(Xukuang's Parameters)
================================================================================
Purpose:
This script trains three U-Net-based architectures for semantic segmentation
using a fixed set of hyperparameters derived from the 'bead_seg.ipynb'
notebook.

Hyperparameters:
- Image Size: 512x512 (grayscale)
- Batch Size: 4
- Epochs: 200
- Optimizer: Adam
- Learning Rate: 0.005
- Loss Function: Binary Focal Loss (gamma=2)
- Architectures:
  1. UNet
  2. Attention_UNet
  3. Attention_ResUNet
- Base Filters: 64 (as per models.py)
- Dropout: 0.0 (as per models.py)

Input:
- Dataset Directory: './dataset_shrunk_masks/'
  - Must contain 'images' and 'masks' subdirectories with .png files.
- Model Definitions: 'models.py'
- Loss Functions: 'loss_functions_fixed.py'

Output:
- A timestamped directory (e.g., 'training_run_20251015_120000/') containing:
  - The best performing model (.hdf5) for each architecture.
  - The full training history (.csv) for each architecture.
  - A final 'summary.md' report detailing the training results.
================================================================================
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
from datetime import datetime
from PIL import Image

# --- Import custom modules ---
# These files must be in the same directory as this script.
try:
    from models import UNet, Attention_UNet, Attention_ResUNet, jacard_coef
    from loss_functions_fixed import BinaryFocalLoss
except ImportError as e:
    print(f"ERROR: Could not import required modules. Make sure 'models.py' and 'loss_functions_fixed.py' are present.")
    print(f"Details: {e}")
    exit(1)

===============================================================================
SETUP AND CONFIGURATION
===============================================================================

--- Training Hyperparameters ---
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1  # Grayscale
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

BATCH_SIZE = 4
EPOCHS = 200
LEARNING_RATE = 5e-3

# --- Paths and Directories ---
DATASET_PATH = './dataset_shrunk_masks/'
IMAGE_DIR = os.path.join(DATASET_PATH, 'images/')
MASK_DIR = os.path.join(DATASET_PATH, 'masks/')

# Create a timestamped directory for this training run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f'training_run_xukuang_{TIMESTAMP}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

===============================================================================
HELPER FUNCTIONS
===============================================================================

def setup_gpu():
    """Configures GPU for memory growth to prevent pre-allocation."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"✗ ERROR setting up GPU: {e}")
    else:
        print("⚠ WARNING: No GPU detected. Training will be very slow.")

def load_data(image_dir, mask_dir, img_height, img_width):
    """Loads and preprocesses image and mask data."""
    print("\n--- Loading and Preprocessing Data ---")
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('.png')])

    if not image_paths or not mask_paths:
        print(f"✗ ERROR: No images or masks found in {image_dir} or {mask_dir}")
        exit(1)
        
    print(f"Found {len(image_paths)} images and {len(mask_paths)} masks.")

    images = np.zeros((len(image_paths), img_height, img_width, 1), dtype=np.float32)
    masks = np.zeros((len(mask_paths), img_height, img_width, 1), dtype=np.float32)

    for i, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_width, img_height))
        images[i] = np.expand_dims(img, axis=-1)

    for i, path in enumerate(mask_paths):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_width, img_height))
        masks[i] = np.expand_dims(mask, axis=-1)

    # Normalize images to [0, 1] and masks to binary {0, 1}
    images = images / 255.0
    masks = masks / 255.0
    
    print("✓ Data loaded and preprocessed.")
    return images, masks

def create_summary_report(results):
    """Creates a markdown summary of the training run."""
    report_path = os.path.join(OUTPUT_DIR, 'summary.md')
    with open(report_path, 'w') as f:
        f.write("# Training Run Summary (Xukuang's Parameters)\n\n")
        f.write(f"**Timestamp**: {TIMESTAMP}\n")
        f.write(f"**Training Script**: `train_models_512_xukuang_parameters.py`\n")
        f.write(f"**PBS Script**: `pbs_train_models_512_xukuang_parameters.sh`\n\n")
        
        f.write("## Hyperparameters\n")
        f.write(f"- **Image Size**: {IMG_HEIGHT}x{IMG_WIDTH}\n")
        f.write(f"- **Epochs**: {EPOCHS}\n")
        f.write(f"- **Batch Size**: {BATCH_SIZE}\n")
        f.write(f"- **Learning Rate**: {LEARNING_RATE}\n")
        f.write(f"- **Optimizer**: Adam\n")
        f.write(f"- **Loss Function**: BinaryFocalLoss(gamma=2)\n\n")

        f.write("## Environment\n")
        f.write(f"- **Input Dataset**: `{DATASET_PATH}`\n")
        f.write(f"- **Output Directory**: `{OUTPUT_DIR}`\n\n")

        f.write("## Training Results\n\n")
        f.write("| Model Architecture | Best val_jacard_coef | Training Time | Saved Model | History File |\n")
        f.write("|--------------------|----------------------|---------------|-------------|--------------|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['best_jaccard']:.4f} | {str(r['time']).split('.')[0]} | `{r['model_path']}` | `{r['csv_path']}` |\n")
        
        f.write("\n\n**WARNING**: The models were trained with a base of 64 filters. This is memory-intensive for 512x512 images and may lead to Out-of-Memory (OOM) errors on some systems. The reference HPC script `pbs_hyperparam_search_512.sh` suggested reducing filters to 32.\n")

    print(f"\n✓ Training summary saved to: {report_path}")

===============================================================================
MAIN TRAINING EXECUTION
===============================================================================

def main():
    """Main function to execute the training pipeline."""
    start_time = datetime.now()
    print("================================================================================")
    print(f"STARTING TRAINING RUN (Xukuang's Parameters) at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("================================================================================")

    setup_gpu()
    
    # --- Load Data ---
    images, masks = load_data(IMAGE_DIR, MASK_DIR, IMG_HEIGHT, IMG_WIDTH)
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)
    print(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # --- Define Models and Training Configurations ---
    models_to_train = [
        {"constructor": UNet, "name": "UNet"},
        {"constructor": Attention_UNet, "name": "Attention_UNet"},
        {"constructor": Attention_ResUNet, "name": "Attention_ResUNet"}
    ]
    
    training_results = []

    for model_config in models_to_train:
        model_name = model_config["name"]
        print(f"\n--- Training Model: {model_name} ---")
        
        # --- Build and Compile Model ---
        model = model_config["constructor"](input_shape=INPUT_SHAPE, dropout_rate=0.0, batch_norm=True)
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                      loss=BinaryFocalLoss(gamma=2),
                      metrics=['accuracy', jacard_coef])
        
        print(f"✓ Model {model_name} compiled.")
        model.summary()

        # --- Callbacks ---
        model_path = os.path.join(OUTPUT_DIR, f'best_{model_name}.hdf5')
        csv_path = os.path.join(OUTPUT_DIR, f'history_{model_name}.csv')
        
        checkpoint = ModelCheckpoint(model_path, monitor='val_jacard_coef', verbose=1, save_best_only=True, mode='max')
        csv_logger = CSVLogger(csv_path)
        
        callbacks_list = [checkpoint, csv_logger]

        # --- Train Model ---
        train_start_time = datetime.now()
        
        history = model.fit(X_train, y_train,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            verbose=1,
                            validation_data=(X_test, y_test),
                            callbacks=callbacks_list,
                            shuffle=False)
        
        train_end_time = datetime.now()
        execution_time = train_end_time - train_start_time
        
        best_jaccard = max(history.history['val_jacard_coef'])
        
        print(f"✓ Training for {model_name} completed in {execution_time}.")
        print(f"✓ Best validation Jaccard coefficient: {best_jaccard:.4f}")

        training_results.append({
            "name": model_name,
            "best_jaccard": best_jaccard,
            "time": execution_time,
            "model_path": f'best_{model_name}.hdf5',
            "csv_path": f'history_{model_name}.csv'
        })

    # --- Finalization ---
    create_summary_report(training_results)
    
    end_time = datetime.now()
    print("\n================================================================================")
    print(f"TRAINING RUN COMPLETED at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {end_time - start_time}")
    print("================================================================================")


if __name__ == "__main__":
    main()
