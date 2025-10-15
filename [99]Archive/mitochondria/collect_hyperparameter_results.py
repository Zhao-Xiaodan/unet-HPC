#!/usr/bin/env python3
"""
Collect hyperparameter optimization results from individual JSON files
and create a clean CSV summary for analysis.
"""

import os
import json
import pandas as pd
import glob

def collect_results(results_dir):
    """Collect all successful experiment results."""

    # Find all JSON results files
    json_files = glob.glob(os.path.join(results_dir, "exp_*", "*_results.json"))

    results = []

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Only include successful experiments
            if data.get('training_successful', False):
                results.append(data)
                print(f"✓ Loaded: {os.path.basename(json_file)}")
            else:
                print(f"✗ Failed: {os.path.basename(json_file)}")

        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    if not results:
        print("No successful experiments found!")
        return None

    # Create DataFrame
    df = pd.DataFrame(results)

    # Clean and sort
    df = df.sort_values(['architecture', 'learning_rate', 'batch_size'])

    print(f"\nCollected {len(df)} successful experiments:")
    print(df[['architecture', 'learning_rate', 'batch_size', 'best_val_jaccard', 'val_loss_stability']].round(4))

    return df

def main():
    results_dir = "hyperparameter_optimization_20250926_123742"

    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found!")
        return

    # Collect results
    df = collect_results(results_dir)

    if df is not None:
        # Save clean CSV
        output_file = os.path.join(results_dir, "hyperparameter_summary_clean.csv")
        df.to_csv(output_file, index=False)
        print(f"\n✓ Clean results saved to: {output_file}")

        # Print summary statistics
        print(f"\nSUMMARY STATISTICS:")
        print(f"Total experiments: {len(df)}")
        print(f"Architectures: {', '.join(df['architecture'].unique())}")
        print(f"Learning rates: {', '.join([str(x) for x in sorted(df['learning_rate'].unique())])}")
        print(f"Batch sizes: {', '.join([str(x) for x in sorted(df['batch_size'].unique())])}")

        # Best results
        best_idx = df['best_val_jaccard'].idxmax()
        best_result = df.loc[best_idx]
        print(f"\nBEST RESULT:")
        print(f"Architecture: {best_result['architecture']}")
        print(f"Learning Rate: {best_result['learning_rate']}")
        print(f"Batch Size: {best_result['batch_size']}")
        print(f"Best Val Jaccard: {best_result['best_val_jaccard']:.4f}")
        print(f"Stability: {best_result['val_loss_stability']:.4f}")

if __name__ == "__main__":
    main()