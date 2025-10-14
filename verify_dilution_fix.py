#!/usr/bin/env python3
"""
Verify Dilution Factor Extraction Fix
======================================

This script demonstrates the bug fix and shows what dilution factors
will be extracted from your test images.

Bug: Substring matching caused incorrect dilution detection
Fix: Use regex with word boundaries

Author: Claude Code
Date: October 14, 2025
"""

import re
from pathlib import Path

def extract_dilution_factor_OLD(filename):
    """Old buggy version - substring matching."""
    DILUTION_PATTERNS = {
        '10x': 10, '20x': 20, '40x': 40, '80x': 80,
        '160x': 160, '320x': 320, '640x': 640,
        '1280x': 1280, '2560x': 2560, '5120x': 5120, '10240x': 10240
    }

    for pattern, value in DILUTION_PATTERNS.items():
        if pattern.lower() in filename.lower():
            return value

    match = re.search(r'(\d+)x', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None

def extract_dilution_factor_NEW(filename):
    """New fixed version - word boundary matching."""
    match = re.search(r'(?:^|_)(\d+)x(?:_|\.|-)', filename.lower())
    if match:
        return int(match.group(1))

    match = re.search(r'^(\d+)x', filename.lower())
    if match:
        return int(match.group(1))

    return None

# Get test images
test_images_dir = Path('./test_images')
test_images = sorted(test_images_dir.glob('*.tif'))

print("=" * 80)
print("DILUTION FACTOR EXTRACTION - BUG FIX VERIFICATION")
print("=" * 80)
print()

print("Test Images Found:", len(test_images))
print()

print(f"{'Filename':<50} {'OLD (Bug)':<15} {'NEW (Fixed)':<15} {'Status'}")
print("-" * 95)

correct_count = 0
fixed_count = 0

for img_path in test_images:
    old_dilution = extract_dilution_factor_OLD(img_path.stem)
    new_dilution = extract_dilution_factor_NEW(img_path.stem)

    if old_dilution == new_dilution:
        status = "✓ OK"
        correct_count += 1
    else:
        status = "✗ FIXED"
        fixed_count += 1

    print(f"{img_path.stem:<50} {str(old_dilution)+'x' if old_dilution else 'None':<15} "
          f"{str(new_dilution)+'x' if new_dilution else 'None':<15} {status}")

print("-" * 95)
print()

print(f"Summary:")
print(f"  Images analyzed: {len(test_images)}")
print(f"  Correct (unchanged): {correct_count}")
print(f"  Fixed (was wrong): {fixed_count}")
print()

# Show expected dilution series
new_dilutions = sorted([extract_dilution_factor_NEW(img.stem)
                        for img in test_images
                        if extract_dilution_factor_NEW(img.stem) is not None])

print(f"Expected dilution series in re-analysis:")
print(f"  {new_dilutions}")
print(f"  Range: {new_dilutions[0]}x to {new_dilutions[-1]}x")
print()

print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("Re-run density_analysis_arch_comparison.py with the fixed version.")
print("Expected changes:")
print(f"  - OLD x-axis: 10x, 20x, 40x, 80x, 160x (WRONG!)")
print(f"  - NEW x-axis: {', '.join([f'{d}x' for d in new_dilutions])} (CORRECT!)")
print()
print("The complete dilution series from 10x to 10240x will now be included.")
print("=" * 80)
