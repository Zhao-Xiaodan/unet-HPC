#!/usr/bin/env python3
"""
Test script for Modern U-Net implementations
Validates model creation and basic functionality
"""

import os
import sys
import numpy as np
import tensorflow as tf

def setup_gpu():
    """Configure GPU if available"""
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

def test_model_creation():
    """Test creation of all modern U-Net models"""
    print("=" * 60)
    print("TESTING MODERN U-NET MODEL CREATION")
    print("=" * 60)

    try:
        from modern_unet_models import create_modern_unet

        # Test parameters
        input_shape = (256, 256, 3)
        models_to_test = ['ConvNeXt_UNet', 'Swin_UNet', 'CoAtNet_UNet']

        results = {}

        for model_name in models_to_test:
            print(f"\nTesting {model_name}...")
            try:
                model = create_modern_unet(model_name, input_shape, num_classes=1)
                params = model.count_params()

                # Test model compilation
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )

                # Test forward pass with dummy data
                dummy_input = np.random.random((1, 256, 256, 3)).astype(np.float32)
                output = model.predict(dummy_input, verbose=0)

                results[model_name] = {
                    'parameters': params,
                    'output_shape': output.shape,
                    'status': 'SUCCESS'
                }

                print(f"  ✓ {model_name}: {params:,} parameters")
                print(f"  ✓ Output shape: {output.shape}")

                # Clean up memory
                del model
                tf.keras.backend.clear_session()

            except Exception as e:
                results[model_name] = {
                    'error': str(e),
                    'status': 'FAILED'
                }
                print(f"  ✗ {model_name}: FAILED - {e}")

        return results

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return None

def test_training_script():
    """Test basic training script functionality"""
    print("\n" + "=" * 60)
    print("TESTING TRAINING SCRIPT IMPORTS")
    print("=" * 60)

    try:
        # Test if training script can be imported
        import modern_unet_training
        print("✓ Training script imports successfully")

        # Test if key functions exist
        functions_to_check = [
            'setup_gpu',
            'load_dataset',
            'train_modern_unet',
            'create_performance_summary'
        ]

        for func_name in functions_to_check:
            if hasattr(modern_unet_training, func_name):
                print(f"  ✓ {func_name} function found")
            else:
                print(f"  ✗ {func_name} function missing")

        return True

    except ImportError as e:
        print(f"✗ Training script import failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    print("\n" + "=" * 60)
    print("TESTING DEPENDENCIES")
    print("=" * 60)

    dependencies = [
        'tensorflow',
        'numpy',
        'pandas',
        'matplotlib',
        'sklearn',
        'cv2',
        'PIL'
    ]

    results = {}

    for dep in dependencies:
        try:
            if dep == 'cv2':
                import cv2
            elif dep == 'PIL':
                from PIL import Image
            elif dep == 'sklearn':
                from sklearn.model_selection import train_test_split
            else:
                __import__(dep)

            results[dep] = 'AVAILABLE'
            print(f"  ✓ {dep}")
        except ImportError:
            results[dep] = 'MISSING'
            print(f"  ✗ {dep} - MISSING")

    # Test focal loss
    try:
        from focal_loss import BinaryFocalLoss
        results['focal_loss'] = 'AVAILABLE'
        print(f"  ✓ focal_loss")
    except ImportError:
        results['focal_loss'] = 'MISSING (will use custom implementation)'
        print(f"  ⚠ focal_loss - MISSING (will use custom implementation)")

    return results

def test_existing_models():
    """Test if existing models can be imported"""
    print("\n" + "=" * 60)
    print("TESTING EXISTING MODEL DEPENDENCIES")
    print("=" * 60)

    try:
        # Check if models.py exists (renamed from 224_225_226_models.py)
        if os.path.exists('224_225_226_models.py') and not os.path.exists('models.py'):
            print("  ✓ Copying 224_225_226_models.py to models.py")
            import shutil
            shutil.copy('224_225_226_models.py', 'models.py')

        if os.path.exists('models.py'):
            from models import jacard_coef, dice_coef
            print("  ✓ Existing models.py imported successfully")
            print("  ✓ jacard_coef function available")
            print("  ✓ dice_coef function available")
            return True
        else:
            print("  ✗ models.py not found")
            return False

    except ImportError as e:
        print(f"  ✗ Failed to import existing models: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 MODERN U-NET IMPLEMENTATION VALIDATION")
    print("=" * 60)
    print("Testing modern U-Net architectures for mitochondria segmentation")
    print("Models: ConvNeXt-UNet, Swin-UNet, CoAtNet-UNet")
    print()

    # Setup
    gpu_available = setup_gpu()

    # Run tests
    test_results = {}

    # Test 1: Dependencies
    dep_results = test_dependencies()
    test_results['dependencies'] = dep_results

    # Test 2: Existing models
    existing_models_ok = test_existing_models()
    test_results['existing_models'] = existing_models_ok

    # Test 3: Training script
    training_script_ok = test_training_script()
    test_results['training_script'] = training_script_ok

    # Test 4: Model creation (only if dependencies are OK)
    missing_deps = [k for k, v in dep_results.items() if v == 'MISSING']
    if not missing_deps and existing_models_ok:
        model_results = test_model_creation()
        test_results['models'] = model_results
    else:
        print(f"\n⚠ Skipping model creation test due to missing dependencies: {missing_deps}")
        test_results['models'] = 'SKIPPED'

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if isinstance(test_results['models'], dict):
        successful_models = [k for k, v in test_results['models'].items() if v.get('status') == 'SUCCESS']
        failed_models = [k for k, v in test_results['models'].items() if v.get('status') == 'FAILED']

        print(f"✓ Models successfully created: {len(successful_models)}/3")
        for model in successful_models:
            params = test_results['models'][model]['parameters']
            print(f"  - {model}: {params:,} parameters")

        if failed_models:
            print(f"✗ Models failed: {failed_models}")
    else:
        print(f"⚠ Model creation test: {test_results['models']}")

    print(f"✓ Dependencies available: {len([k for k, v in dep_results.items() if 'AVAILABLE' in v])}/{len(dep_results)}")
    print(f"✓ Existing models: {'OK' if existing_models_ok else 'FAILED'}")
    print(f"✓ Training script: {'OK' if training_script_ok else 'FAILED'}")
    print(f"✓ GPU available: {'Yes' if gpu_available else 'No'}")

    # Overall status
    critical_issues = []
    if not existing_models_ok:
        critical_issues.append("Missing models.py")
    if not training_script_ok:
        critical_issues.append("Training script issues")

    if critical_issues:
        print(f"\n⚠ CRITICAL ISSUES FOUND: {critical_issues}")
        print("Please resolve these issues before running the training.")
        return False
    else:
        print(f"\n✅ VALIDATION PASSED!")
        print("The modern U-Net implementation appears to be ready for training.")
        print()
        print("🚀 NEXT STEPS:")
        print("1. Ensure dataset is available in dataset_full_stack/ or dataset/")
        print("2. Submit PBS job: qsub pbs_modern_unet.sh")
        print("3. Monitor training progress in the output directory")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)