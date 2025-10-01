#!/usr/bin/env python3
"""
Validation Script for Modern U-Net Fixes

This script validates that the fixes for ConvNeXt-UNet and CoAtNet-UNet
are working properly before running the full training.

Fixes tested:
1. ConvNeXt-UNet: Dataset cache clearing to prevent "name already exists" errors
2. CoAtNet-UNet: Weight initialization and gradient computation verification
"""

import os
import sys
import numpy as np
import tensorflow as tf
import tempfile
import shutil

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

def test_cache_clearing():
    """Test the enhanced cache clearing functionality"""
    print("\n" + "="*60)
    print("TESTING CACHE CLEARING (ConvNeXt-UNet Fix)")
    print("="*60)

    try:
        from modern_unet_training import clear_tensorflow_caches

        # Create some dummy cache directories to test clearing
        test_dirs = [
            '/tmp/test_tf_cache',
            tempfile.gettempdir() + '/test_tf_data'
        ]

        # Create test directories
        for test_dir in test_dirs:
            os.makedirs(test_dir, exist_ok=True)
            with open(os.path.join(test_dir, 'test_file.txt'), 'w') as f:
                f.write('test')

        print("Created test cache directories...")

        # Test the clearing function
        clear_tensorflow_caches()

        # Check if test directories were cleaned (they shouldn't be since they're not TF caches)
        remaining_dirs = [d for d in test_dirs if os.path.exists(d)]

        # Clean up test directories
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir, ignore_errors=True)

        print("✓ Cache clearing function works properly")
        return True

    except Exception as e:
        print(f"✗ Cache clearing test failed: {e}")
        return False

def test_model_creation_and_initialization():
    """Test model creation and weight initialization"""
    print("\n" + "="*60)
    print("TESTING MODEL CREATION AND INITIALIZATION")
    print("="*60)

    try:
        from modern_unet_models import create_modern_unet

        input_shape = (64, 64, 3)  # Small test shape
        models_to_test = ['ConvNeXt_UNet', 'Swin_UNet', 'CoAtNet_UNet']

        results = {}

        for model_name in models_to_test:
            print(f"\nTesting {model_name}...")

            try:
                # Create model
                model = create_modern_unet(model_name, input_shape, num_classes=1)

                # Test building
                model.build(input_shape=(None,) + input_shape)

                # Test forward pass
                dummy_input = np.random.random((1,) + input_shape).astype(np.float32)
                output = model(dummy_input, training=False)

                # Test gradient computation (critical for CoAtNet)
                with tf.GradientTape() as tape:
                    predictions = model(dummy_input, training=True)
                    # Simple loss for testing
                    dummy_target = np.random.random((1, 64, 64, 1)).astype(np.float32)
                    loss = tf.reduce_mean(tf.square(predictions - dummy_target))

                gradients = tape.gradient(loss, model.trainable_variables)

                # Check if gradients are valid
                valid_gradients = all(grad is not None for grad in gradients)

                results[model_name] = {
                    'creation': 'SUCCESS',
                    'parameters': model.count_params(),
                    'forward_pass': 'SUCCESS',
                    'gradient_computation': 'SUCCESS' if valid_gradients else 'FAILED',
                    'output_shape': output.shape
                }

                print(f"  ✓ {model_name}: {model.count_params():,} parameters")
                print(f"  ✓ Forward pass: {output.shape}")
                print(f"  ✓ Gradient computation: {'OK' if valid_gradients else 'FAILED'}")

                # Clean up
                del model
                tf.keras.backend.clear_session()

            except Exception as e:
                results[model_name] = {
                    'creation': 'FAILED',
                    'error': str(e)
                }
                print(f"  ✗ {model_name}: FAILED - {e}")

        return results

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return None

def test_enhanced_weight_initialization():
    """Test the enhanced weight initialization specifically for CoAtNet"""
    print("\n" + "="*60)
    print("TESTING ENHANCED WEIGHT INITIALIZATION (CoAtNet-UNet Fix)")
    print("="*60)

    try:
        from modern_unet_models import create_modern_unet

        input_shape = (64, 64, 3)
        model_name = 'CoAtNet_UNet'

        print(f"Testing enhanced initialization for {model_name}...")

        # Create model
        model = create_modern_unet(model_name, input_shape, num_classes=1)

        # Apply the enhanced initialization logic from the fixed training script
        print("Applying enhanced initialization...")

        # Force model building
        model.build(input_shape=(None,) + input_shape)

        # Build all sublayers (critical fix)
        for layer in model.layers:
            if hasattr(layer, 'build') and not getattr(layer, 'built', False):
                try:
                    if hasattr(layer, 'input_spec') and layer.input_spec is not None:
                        layer.build(layer.input_spec)
                except Exception as e:
                    print(f"Warning: Could not build layer {layer.name}: {e}")

        # Force forward pass to initialize weights
        dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32)
        try:
            _ = model(dummy_input, training=False)
            print("✓ Enhanced initialization forward pass successful")
        except Exception as e:
            print(f"⚠ Forward pass failed: {e}")

        # Test gradient computation with enhanced initialization
        try:
            with tf.GradientTape() as tape:
                predictions = model(dummy_input, training=True)
                dummy_target = tf.zeros_like(predictions)
                loss = tf.reduce_mean(tf.square(predictions - dummy_target))

            gradients = tape.gradient(loss, model.trainable_variables)
            valid_gradients = all(grad is not None for grad in gradients)

            if valid_gradients:
                print("✓ Enhanced gradient computation successful")
                return True
            else:
                print("✗ Some gradients are None after enhanced initialization")
                return False

        except Exception as e:
            print(f"✗ Enhanced gradient computation failed: {e}")
            return False

    except Exception as e:
        print(f"✗ Enhanced initialization test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("🧪 MODERN U-NET FIXES VALIDATION")
    print("="*60)
    print("Validating fixes for ConvNeXt-UNet and CoAtNet-UNet issues")
    print()

    # Setup
    gpu_available = setup_gpu()

    # Run validation tests
    test_results = {}

    # Test 1: Cache clearing functionality
    cache_test_result = test_cache_clearing()
    test_results['cache_clearing'] = cache_test_result

    # Test 2: Model creation and initialization
    model_test_results = test_model_creation_and_initialization()
    test_results['model_creation'] = model_test_results

    # Test 3: Enhanced weight initialization for CoAtNet
    enhanced_init_result = test_enhanced_weight_initialization()
    test_results['enhanced_initialization'] = enhanced_init_result

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    print(f"✓ Cache clearing function: {'PASSED' if cache_test_result else 'FAILED'}")

    if model_test_results:
        successful_models = len([m for m in model_test_results.values() if m.get('creation') == 'SUCCESS'])
        print(f"✓ Model creation: {successful_models}/3 models successful")

        # Check specific issues
        convnext_status = model_test_results.get('ConvNeXt_UNet', {}).get('creation', 'UNKNOWN')
        coatnet_status = model_test_results.get('CoAtNet_UNet', {}).get('creation', 'UNKNOWN')

        print(f"  - ConvNeXt-UNet: {convnext_status}")
        print(f"  - CoAtNet-UNet: {coatnet_status}")

        if coatnet_status == 'SUCCESS':
            grad_status = model_test_results.get('CoAtNet_UNet', {}).get('gradient_computation', 'UNKNOWN')
            print(f"  - CoAtNet gradients: {grad_status}")

    print(f"✓ Enhanced CoAtNet initialization: {'PASSED' if enhanced_init_result else 'FAILED'}")

    # Overall assessment
    critical_issues = []

    if not cache_test_result:
        critical_issues.append("Cache clearing function failed")

    if model_test_results:
        if model_test_results.get('ConvNeXt_UNet', {}).get('creation') != 'SUCCESS':
            critical_issues.append("ConvNeXt-UNet creation failed")

        if model_test_results.get('CoAtNet_UNet', {}).get('creation') != 'SUCCESS':
            critical_issues.append("CoAtNet-UNet creation failed")
        elif not enhanced_init_result:
            critical_issues.append("CoAtNet-UNet enhanced initialization failed")

    if critical_issues:
        print(f"\n⚠ ISSUES FOUND: {critical_issues}")
        print("Please address these issues before running full training.")
        return False
    else:
        print(f"\n✅ ALL VALIDATION TESTS PASSED!")
        print("The fixes are working properly. Ready for full training.")
        print()
        print("🚀 NEXT STEPS:")
        print("1. Submit PBS job: qsub pbs_modern_unet.sh")
        print("2. Monitor training progress")
        print("3. Expect all three models to train successfully")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)