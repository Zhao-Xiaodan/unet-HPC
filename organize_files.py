#!/usr/bin/env python3
"""
Organize training scripts and outputs into their respective directories.

Moves .py, .sh, .md, and .log files into their corresponding output directories.
"""

import os
import shutil
from pathlib import Path
import re

# Mapping of output directories to their source files
FILE_MAPPINGS = {
    # Density experiments (already partially organized)
    'density_analysis_arch_comparison_20251014_004358': [
        'density_analysis_arch_comparison.py',
        'pbs_density_analysis.sh',
    ],
    'density_prediction_256_20251014_054939': [
        'density_prediction_256_fast.py',
        'pbs_density_256_fast.sh',
    ],

    # Hyperparameter searches
    'hyperparameter_search_20251013_154754': [
        'hyperparameter_search_residual_architectures.py',
        'pbs_hyperparameter_search_residual.sh',
        'reanalyze_hyperparameter_search.py',
        'generate_updated_report.py',
        'HYPERPARAM_REANALYSIS_SUMMARY.md',
    ],
    'hyperparameter_optimization_20250927_101211': [
        'hyperparameter_optimization.py',
        'pbs_hyperparameter_optimization.sh',
        'analyze_hyperparameter_results.py',
    ],
    'hyperparameter_optimization_20250926_165036': [
        # Earlier hyperparameter run (if has unique scripts)
    ],
    'hyperparameter_optimization_20250926_123742': [
        # Earliest hyperparameter run
    ],

    # Validation experiments
    'validation_arch_comparison_20251013_093844': [
        'validation_architecture_comparison.py',
        'pbs_validation_arch_comparison.sh',
        'analyze_architecture_comparison_5fold.py',
        'create_arch_comparison_report.py',
    ],
    'validation_cv_20251013_052113': [
        'validation_cv_training.py',
        'pbs_validation_cv.sh',
    ],
    'validation_fixes_20251012_234806': [
        'validation_fixes_training.py',
        'pbs_validation_fixes.sh',
    ],
    'validation_focal_tversky_20251013_001124': [
        'validation_focal_tversky.py',
        'pbs_validation_focal_tversky.sh',
    ],
    'validation_small_model_20251013_050005': [
        'validation_small_model.py',
        'pbs_validation_small_model.sh',
    ],
    'validation_analysis_20250929_100953': [
        'analyze_broken_vs_fixed_validation.py',
        'analyze_fixed_results_comparison.py',
    ],

    # Dataset studies
    'dataset_size_study_20250929_110609': [
        '224_225_226_dataset_size_study.py',
        'pbs_dataset_size_study.sh',
    ],

    # Architecture experiments
    'mitochondria_segmentation_20250925_133928': [
        '224_225_226_mito_segm_using_various_unet_models.py',
        '224_225_226_models.py',
        'pbs_unet.sh',
        'analyze_unet_comparison.py',
    ],
    'mitochondria_segmentation_original_20250928_210433': [
        '224_225_226_mito_segm_using_various_unet_models_original.py',
        '224_225_226_models_original.py',
    ],

    # Modern architectures
    'modern_unet_training_20251001_120110': [
        'modern_unet_optimized_training.py',
        'pbs_modern_unet_optimized.sh',
    ],
    'modern_unet_training_20251001_040132': [
        'modern_unet_training.py',
        'pbs_modern_unet.sh',
    ],
    'convnext_unet_training_20251002_093834': [
        'convnext_unet_training.py',
        'pbs_convnext_unet.sh',
        'convnext_unet_optimized_training.py',
        'ConvNeXt_CoAtNet_Optimization_Summary.md',
    ],
    'coatnet_unet_training_20251001_155445': [
        'coatnet_unet_training.py',
        'pbs_coatnet_unet.sh',
    ],

    # Microbead experiments
    'microbead_training_20251009_073134': [
        'microbead_training.py',
        'pbs_microbead_training.sh',
        'analyze_microbead_training_results.py',
        'analyze_microbead_dataset.py',
    ],

    # Microscope experiments
    'microscope_training_20251008_074915': [
        'microscope_training.py',
        'pbs_microscope_training.sh',
        'analyze_microscope_results.py',
    ],

    # Comprehensive hyperparameter searches
    'hyperparam_comprehensive_20251012_005054': [
        'hyperparam_comprehensive_search.py',
        'pbs_hyperparam_comprehensive.sh',
        'analyze_comprehensive_search.py',
    ],
    'hyperparam_comprehensive_20251011_031111': [
        # Earlier comprehensive search
    ],
    'hyperparam_search_20251010_043123': [
        'hyperparam_search_focused.py',
        'pbs_hyperparam_search.sh',
        'analyze_hyperparam_search_results.py',
    ],

    # Prediction analysis
    'prediction_analysis_20251012_074415': [
        'prediction_comparison_analysis.py',
        'pbs_prediction_analysis.sh',
    ],
    'predictions_20251010_110029': [
        'generate_predictions.py',
        'pbs_predictions.sh',
    ],
    'predictions_20251009_140645': [
        # Earlier predictions
    ],

    # Analysis directories
    'breakthrough_analysis_20250928': [
        'analyze_breakthrough.py',
    ],
    'training_analysis_20250927': [
        'analyze_training_results.py',
        'analyze_fixed_training_comparison.py',
    ],
}

# Find log files and match them to directories
def find_matching_logs(directory_name):
    """Find log files that match a directory timestamp pattern."""
    # Extract timestamp from directory name (YYYYMMDD_HHMMSS or YYYYMMDD)
    timestamp_match = re.search(r'(\d{8}_\d{6}|\d{8})', directory_name)
    if not timestamp_match:
        return []

    timestamp = timestamp_match.group(1)

    # Find log files with similar timestamp
    log_files = []
    for f in Path('.').glob('*.log'):
        if timestamp in f.name or directory_name.split('_')[0] in f.name:
            log_files.append(f.name)

    return log_files

def organize_files(dry_run=True):
    """Organize files into their respective directories."""
    print("="*80)
    print("FILE ORGANIZATION SCRIPT")
    print("="*80)
    print(f"Mode: {'DRY RUN (no files moved)' if dry_run else 'LIVE (files will be moved)'}")
    print()

    moved_files = []
    skipped_files = []

    for directory, files in FILE_MAPPINGS.items():
        dir_path = Path(directory)

        # Check if directory exists
        if not dir_path.exists():
            print(f"⚠ Directory not found: {directory}")
            continue

        # Add log files
        log_files = find_matching_logs(directory)
        all_files = list(files) + log_files

        if not all_files:
            continue

        print(f"\n📁 {directory}/")

        for filename in all_files:
            source = Path(filename)

            if not source.exists():
                # print(f"  ⊘ {filename} (not found in root)")
                continue

            # Check if already in directory
            dest = dir_path / filename
            if dest.exists():
                print(f"  ✓ {filename} (already exists)")
                skipped_files.append(filename)
                continue

            # Move or report
            if dry_run:
                print(f"  → {filename} (would move)")
            else:
                try:
                    shutil.copy2(source, dest)
                    print(f"  ✓ {filename} (copied)")
                    moved_files.append(filename)
                except Exception as e:
                    print(f"  ✗ {filename} (error: {e})")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if dry_run:
        print(f"Files that would be moved: {len([f for files in FILE_MAPPINGS.values() for f in files if Path(f).exists()])}")
        print("\nTo actually move files, run:")
        print("  python3 organize_files.py --execute")
    else:
        print(f"Files copied: {len(moved_files)}")
        print(f"Files skipped (already exist): {len(skipped_files)}")
        print("\nTo remove originals from root, manually delete them after verification.")

    print("="*80)

if __name__ == '__main__':
    import sys

    # Check if --execute flag provided
    execute = '--execute' in sys.argv or '-x' in sys.argv

    organize_files(dry_run=not execute)
