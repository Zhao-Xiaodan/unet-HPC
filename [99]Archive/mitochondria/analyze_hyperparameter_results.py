#!/usr/bin/env python3
"""
Advanced hyperparameter optimization results analysis script.
Provides comprehensive analysis, statistical testing, and visualization
of U-Net architecture hyperparameter optimization results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import json
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

class HyperparameterAnalyzer:
    """Comprehensive hyperparameter optimization results analyzer."""

    def __init__(self, summary_file, output_dir):
        self.summary_file = summary_file
        self.output_dir = output_dir
        self.df = None
        self.results_summary = {}

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def load_and_clean_data(self):
        """Load results and perform data cleaning."""
        print("Loading and cleaning hyperparameter results...")

        # Load raw data
        raw_df = pd.read_csv(self.summary_file)
        print(f"Raw data: {len(raw_df)} experiments")

        # Filter successful experiments
        self.df = raw_df[raw_df['training_successful'] == True].copy()
        print(f"Successful experiments: {len(self.df)}")

        if len(self.df) == 0:
            raise ValueError("No successful experiments found!")

        # Data type conversions
        numeric_columns = ['learning_rate', 'batch_size', 'best_val_jaccard',
                          'val_loss_stability', 'final_val_loss', 'overfitting_gap']

        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Remove any rows with NaN in critical columns
        critical_columns = ['architecture', 'learning_rate', 'batch_size', 'best_val_jaccard']
        self.df = self.df.dropna(subset=critical_columns)

        print(f"Final cleaned data: {len(self.df)} experiments")
        print(f"Architectures: {', '.join(self.df['architecture'].unique())}")
        print(f"Learning rates: {', '.join([str(x) for x in sorted(self.df['learning_rate'].unique())])}")
        print(f"Batch sizes: {', '.join([str(x) for x in sorted(self.df['batch_size'].unique())])}")

        return self.df

    def statistical_analysis(self):
        """Perform statistical analysis of hyperparameter effects."""
        print("\nPerforming statistical analysis...")

        stats_results = {}

        # 1. ANOVA for hyperparameter significance
        try:
            from scipy.stats import f_oneway

            # Learning rate effect
            lr_groups = [self.df[self.df['learning_rate'] == lr]['best_val_jaccard'].values
                        for lr in self.df['learning_rate'].unique()]
            lr_f_stat, lr_p_value = f_oneway(*lr_groups)

            # Batch size effect
            bs_groups = [self.df[self.df['batch_size'] == bs]['best_val_jaccard'].values
                        for bs in self.df['batch_size'].unique()]
            bs_f_stat, bs_p_value = f_oneway(*bs_groups)

            # Architecture effect
            arch_groups = [self.df[self.df['architecture'] == arch]['best_val_jaccard'].values
                          for arch in self.df['architecture'].unique()]
            arch_f_stat, arch_p_value = f_oneway(*arch_groups)

            stats_results['anova'] = {
                'learning_rate': {'f_stat': lr_f_stat, 'p_value': lr_p_value},
                'batch_size': {'f_stat': bs_f_stat, 'p_value': bs_p_value},
                'architecture': {'f_stat': arch_f_stat, 'p_value': arch_p_value}
            }

        except Exception as e:
            print(f"ANOVA analysis failed: {e}")
            stats_results['anova'] = None

        # 2. Correlation analysis
        numeric_cols = ['learning_rate', 'batch_size', 'best_val_jaccard', 'val_loss_stability']
        correlation_matrix = self.df[numeric_cols].corr()
        stats_results['correlations'] = correlation_matrix

        # 3. Effect sizes (Cohen's d)
        effect_sizes = {}
        try:
            def cohens_d(x, y):
                nx, ny = len(x), len(y)
                dof = nx + ny - 2
                pooled_std = np.sqrt(((nx-1)*x.var() + (ny-1)*y.var()) / dof)
                return (x.mean() - y.mean()) / pooled_std

            # Compare extreme learning rates
            lr_values = sorted(self.df['learning_rate'].unique())
            if len(lr_values) >= 2:
                low_lr_data = self.df[self.df['learning_rate'] == lr_values[0]]['best_val_jaccard']
                high_lr_data = self.df[self.df['learning_rate'] == lr_values[-1]]['best_val_jaccard']
                if len(low_lr_data) > 0 and len(high_lr_data) > 0:
                    effect_sizes['lr_extreme'] = cohens_d(high_lr_data, low_lr_data)

            # Compare extreme batch sizes
            bs_values = sorted(self.df['batch_size'].unique())
            if len(bs_values) >= 2:
                low_bs_data = self.df[self.df['batch_size'] == bs_values[0]]['best_val_jaccard']
                high_bs_data = self.df[self.df['batch_size'] == bs_values[-1]]['best_val_jaccard']
                if len(low_bs_data) > 0 and len(high_bs_data) > 0:
                    effect_sizes['bs_extreme'] = cohens_d(high_bs_data, low_bs_data)

            stats_results['effect_sizes'] = effect_sizes

        except Exception as e:
            print(f"Effect size analysis failed: {e}")
            stats_results['effect_sizes'] = {}

        self.results_summary['statistics'] = stats_results
        return stats_results

    def create_performance_heatmaps(self):
        """Create detailed heatmaps for each architecture."""
        print("Creating performance heatmaps...")

        architectures = self.df['architecture'].unique()
        fig, axes = plt.subplots(2, len(architectures), figsize=(6*len(architectures), 10))
        if len(architectures) == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle('Hyperparameter Optimization Results - Detailed Heatmaps',
                     fontsize=16, fontweight='bold')

        for i, arch in enumerate(architectures):
            arch_data = self.df[self.df['architecture'] == arch]

            # Performance heatmap
            pivot_jaccard = arch_data.pivot_table(
                values='best_val_jaccard',
                index='learning_rate',
                columns='batch_size',
                aggfunc='mean'
            )

            im1 = axes[0, i].imshow(pivot_jaccard.values, cmap='viridis', aspect='auto')
            axes[0, i].set_title(f'{arch}\nBest Validation Jaccard', fontweight='bold')
            axes[0, i].set_xlabel('Batch Size')
            axes[0, i].set_ylabel('Learning Rate')

            # Set ticks and labels
            axes[0, i].set_xticks(range(len(pivot_jaccard.columns)))
            axes[0, i].set_xticklabels(pivot_jaccard.columns)
            axes[0, i].set_yticks(range(len(pivot_jaccard.index)))
            axes[0, i].set_yticklabels([f'{x:.0e}' for x in pivot_jaccard.index])

            # Add text annotations
            for ii in range(len(pivot_jaccard.index)):
                for jj in range(len(pivot_jaccard.columns)):
                    if not np.isnan(pivot_jaccard.iloc[ii, jj]):
                        axes[0, i].text(jj, ii, f'{pivot_jaccard.iloc[ii, jj]:.3f}',
                                       ha='center', va='center', fontweight='bold',
                                       color='white' if pivot_jaccard.iloc[ii, jj] < pivot_jaccard.values.mean() else 'black')

            plt.colorbar(im1, ax=axes[0, i])

            # Stability heatmap
            if 'val_loss_stability' in arch_data.columns:
                pivot_stability = arch_data.pivot_table(
                    values='val_loss_stability',
                    index='learning_rate',
                    columns='batch_size',
                    aggfunc='mean'
                )

                im2 = axes[1, i].imshow(pivot_stability.values, cmap='viridis_r', aspect='auto')
                axes[1, i].set_title(f'{arch}\nTraining Stability (Lower=Better)', fontweight='bold')
                axes[1, i].set_xlabel('Batch Size')
                axes[1, i].set_ylabel('Learning Rate')

                axes[1, i].set_xticks(range(len(pivot_stability.columns)))
                axes[1, i].set_xticklabels(pivot_stability.columns)
                axes[1, i].set_yticks(range(len(pivot_stability.index)))
                axes[1, i].set_yticklabels([f'{x:.0e}' for x in pivot_stability.index])

                # Add text annotations
                for ii in range(len(pivot_stability.index)):
                    for jj in range(len(pivot_stability.columns)):
                        if not np.isnan(pivot_stability.iloc[ii, jj]):
                            axes[1, i].text(jj, ii, f'{pivot_stability.iloc[ii, jj]:.3f}',
                                           ha='center', va='center', fontweight='bold',
                                           color='white' if pivot_stability.iloc[ii, jj] > pivot_stability.values.mean() else 'black')

                plt.colorbar(im2, ax=axes[1, i])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'detailed_heatmaps.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_comparative_analysis(self):
        """Create comprehensive comparative analysis plots."""
        print("Creating comparative analysis plots...")

        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Comprehensive Hyperparameter Analysis', fontsize=16, fontweight='bold')

        # 1. Performance vs Learning Rate (with confidence intervals)
        for arch in self.df['architecture'].unique():
            arch_data = self.df[self.df['architecture'] == arch]
            lr_stats = arch_data.groupby('learning_rate')['best_val_jaccard'].agg(['mean', 'std', 'count'])

            axes[0, 0].errorbar(lr_stats.index, lr_stats['mean'],
                               yerr=lr_stats['std'],
                               marker='o', label=arch, capsize=5)

        axes[0, 0].set_xlabel('Learning Rate')
        axes[0, 0].set_ylabel('Best Val Jaccard (Mean ± Std)')
        axes[0, 0].set_title('Performance vs Learning Rate')
        axes[0, 0].legend()
        axes[0, 0].set_xscale('log')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Performance vs Batch Size
        for arch in self.df['architecture'].unique():
            arch_data = self.df[self.df['architecture'] == arch]
            bs_stats = arch_data.groupby('batch_size')['best_val_jaccard'].agg(['mean', 'std'])

            axes[0, 1].errorbar(bs_stats.index, bs_stats['mean'],
                               yerr=bs_stats['std'],
                               marker='o', label=arch, capsize=5)

        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Best Val Jaccard (Mean ± Std)')
        axes[0, 1].set_title('Performance vs Batch Size')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Stability vs Learning Rate
        if 'val_loss_stability' in self.df.columns:
            for arch in self.df['architecture'].unique():
                arch_data = self.df[self.df['architecture'] == arch]
                lr_stability = arch_data.groupby('learning_rate')['val_loss_stability'].agg(['mean', 'std'])

                axes[1, 0].errorbar(lr_stability.index, lr_stability['mean'],
                                   yerr=lr_stability['std'],
                                   marker='o', label=arch, capsize=5)

            axes[1, 0].set_xlabel('Learning Rate')
            axes[1, 0].set_ylabel('Val Loss Stability (Lower=Better)')
            axes[1, 0].set_title('Training Stability vs Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].set_xscale('log')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Stability vs Batch Size
        if 'val_loss_stability' in self.df.columns:
            for arch in self.df['architecture'].unique():
                arch_data = self.df[self.df['architecture'] == arch]
                bs_stability = arch_data.groupby('batch_size')['val_loss_stability'].agg(['mean', 'std'])

                axes[1, 1].errorbar(bs_stability.index, bs_stability['mean'],
                                   yerr=bs_stability['std'],
                                   marker='o', label=arch, capsize=5)

            axes[1, 1].set_xlabel('Batch Size')
            axes[1, 1].set_ylabel('Val Loss Stability (Lower=Better)')
            axes[1, 1].set_title('Training Stability vs Batch Size')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # 5. Performance vs Stability Scatter
        colors = ['blue', 'orange', 'green']
        for i, arch in enumerate(self.df['architecture'].unique()):
            arch_data = self.df[self.df['architecture'] == arch]
            if 'val_loss_stability' in arch_data.columns:
                axes[2, 0].scatter(arch_data['val_loss_stability'], arch_data['best_val_jaccard'],
                                  c=colors[i % len(colors)], label=arch, alpha=0.7, s=60)

        axes[2, 0].set_xlabel('Val Loss Stability (Lower=Better)')
        axes[2, 0].set_ylabel('Best Val Jaccard')
        axes[2, 0].set_title('Performance vs Stability Trade-off')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # 6. Box plots for architecture comparison
        arch_performance = [self.df[self.df['architecture'] == arch]['best_val_jaccard'].values
                           for arch in self.df['architecture'].unique()]

        box_plot = axes[2, 1].boxplot(arch_performance, labels=self.df['architecture'].unique(), patch_artist=True)

        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[2, 1].set_ylabel('Best Val Jaccard')
        axes[2, 1].set_title('Architecture Performance Distribution')
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def find_optimal_configurations(self):
        """Find and analyze optimal configurations for each architecture."""
        print("Finding optimal configurations...")

        optimal_configs = {}

        for arch in self.df['architecture'].unique():
            arch_data = self.df[self.df['architecture'] == arch]

            # Best overall performance
            best_perf_idx = arch_data['best_val_jaccard'].idxmax()
            best_perf_config = arch_data.loc[best_perf_idx]

            # Best stability (if available)
            best_stability_config = None
            if 'val_loss_stability' in arch_data.columns:
                best_stab_idx = arch_data['val_loss_stability'].idxmin()
                best_stability_config = arch_data.loc[best_stab_idx]

            # Best balanced (performance + stability)
            balanced_config = None
            if 'val_loss_stability' in arch_data.columns:
                # Normalize both metrics to 0-1 range
                norm_perf = (arch_data['best_val_jaccard'] - arch_data['best_val_jaccard'].min()) / \
                           (arch_data['best_val_jaccard'].max() - arch_data['best_val_jaccard'].min())

                norm_stab = 1 - (arch_data['val_loss_stability'] - arch_data['val_loss_stability'].min()) / \
                            (arch_data['val_loss_stability'].max() - arch_data['val_loss_stability'].min())

                combined_score = (norm_perf + norm_stab) / 2
                balanced_idx = arch_data.index[combined_score.idxmax()]
                balanced_config = arch_data.loc[balanced_idx]

            optimal_configs[arch] = {
                'best_performance': best_perf_config,
                'best_stability': best_stability_config,
                'best_balanced': balanced_config
            }

        self.results_summary['optimal_configurations'] = optimal_configs
        return optimal_configs

    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        print("Generating comprehensive analysis report...")

        report_lines = []

        # Header
        report_lines.append("="*80)
        report_lines.append("COMPREHENSIVE HYPERPARAMETER OPTIMIZATION ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total successful experiments: {len(self.df)}")
        report_lines.append("")

        # Dataset summary
        report_lines.append("DATASET SUMMARY:")
        report_lines.append("-" * 20)
        report_lines.append(f"Architectures: {', '.join(self.df['architecture'].unique())}")
        report_lines.append(f"Learning rates: {', '.join([str(x) for x in sorted(self.df['learning_rate'].unique())])}")
        report_lines.append(f"Batch sizes: {', '.join([str(x) for x in sorted(self.df['batch_size'].unique())])}")
        report_lines.append("")

        # Statistical significance
        if 'statistics' in self.results_summary and self.results_summary['statistics']['anova']:
            report_lines.append("STATISTICAL SIGNIFICANCE ANALYSIS (ANOVA):")
            report_lines.append("-" * 45)
            anova_results = self.results_summary['statistics']['anova']

            for factor, results in anova_results.items():
                significance = "SIGNIFICANT" if results['p_value'] < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"{factor.replace('_', ' ').title()}:")
                report_lines.append(f"  F-statistic: {results['f_stat']:.4f}")
                report_lines.append(f"  p-value: {results['p_value']:.6f} ({significance})")
            report_lines.append("")

        # Overall best configuration
        best_overall_idx = self.df['best_val_jaccard'].idxmax()
        best_overall = self.df.loc[best_overall_idx]

        report_lines.append("BEST OVERALL CONFIGURATION:")
        report_lines.append("-" * 30)
        report_lines.append(f"Architecture: {best_overall['architecture']}")
        report_lines.append(f"Learning Rate: {best_overall['learning_rate']}")
        report_lines.append(f"Batch Size: {best_overall['batch_size']}")
        report_lines.append(f"Best Val Jaccard: {best_overall['best_val_jaccard']:.4f}")
        if 'val_loss_stability' in best_overall:
            report_lines.append(f"Stability: {best_overall['val_loss_stability']:.4f}")
        report_lines.append("")

        # Architecture-specific recommendations
        if 'optimal_configurations' in self.results_summary:
            report_lines.append("ARCHITECTURE-SPECIFIC OPTIMAL CONFIGURATIONS:")
            report_lines.append("-" * 50)

            for arch, configs in self.results_summary['optimal_configurations'].items():
                report_lines.append(f"\n{arch}:")
                report_lines.append("  Best Performance Configuration:")
                best_perf = configs['best_performance']
                report_lines.append(f"    Learning Rate: {best_perf['learning_rate']}")
                report_lines.append(f"    Batch Size: {best_perf['batch_size']}")
                report_lines.append(f"    Val Jaccard: {best_perf['best_val_jaccard']:.4f}")

                if configs['best_stability'] is not None:
                    report_lines.append("  Best Stability Configuration:")
                    best_stab = configs['best_stability']
                    report_lines.append(f"    Learning Rate: {best_stab['learning_rate']}")
                    report_lines.append(f"    Batch Size: {best_stab['batch_size']}")
                    report_lines.append(f"    Stability: {best_stab['val_loss_stability']:.4f}")

        # Hyperparameter insights
        report_lines.append("")
        report_lines.append("HYPERPARAMETER INSIGHTS:")
        report_lines.append("-" * 25)

        # Learning rate analysis
        lr_analysis = self.df.groupby('learning_rate')['best_val_jaccard'].agg(['mean', 'std', 'count'])
        report_lines.append("Learning Rate Effects:")
        for lr, row in lr_analysis.iterrows():
            report_lines.append(f"  {lr}: {row['mean']:.4f} ± {row['std']:.4f} (n={row['count']})")

        # Batch size analysis
        bs_analysis = self.df.groupby('batch_size')['best_val_jaccard'].agg(['mean', 'std', 'count'])
        report_lines.append("\nBatch Size Effects:")
        for bs, row in bs_analysis.iterrows():
            report_lines.append(f"  {bs}: {row['mean']:.4f} ± {row['std']:.4f} (n={row['count']})")

        # Architecture comparison
        arch_analysis = self.df.groupby('architecture')['best_val_jaccard'].agg(['mean', 'std', 'count'])
        report_lines.append("\nArchitecture Performance:")
        for arch, row in arch_analysis.iterrows():
            report_lines.append(f"  {arch}: {row['mean']:.4f} ± {row['std']:.4f} (n={row['count']})")

        # Practical recommendations
        report_lines.append("")
        report_lines.append("PRACTICAL RECOMMENDATIONS:")
        report_lines.append("-" * 30)
        report_lines.append("1. For maximum performance: Use best overall configuration")
        report_lines.append("2. For stable training: Prefer larger batch sizes (16-32)")
        report_lines.append("3. For balanced approach: Use learning rates 5e-4 to 1e-3")
        report_lines.append("4. Architecture selection: Consider task-specific requirements")

        # Future work suggestions
        report_lines.append("")
        report_lines.append("FUTURE WORK RECOMMENDATIONS:")
        report_lines.append("-" * 35)
        report_lines.append("1. Extended training (100+ epochs) with optimal configurations")
        report_lines.append("2. Cross-validation to confirm hyperparameter robustness")
        report_lines.append("3. Transfer learning assessment on other datasets")
        report_lines.append("4. Ensemble methods using multiple optimal configurations")

        report_lines.append("")
        report_lines.append("="*80)

        # Save report
        report_path = os.path.join(self.output_dir, 'comprehensive_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        # Print to console
        print('\n'.join(report_lines))

        return report_path

    def save_results_summary(self):
        """Save complete results summary as JSON."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif pd.isna(obj):
                return None
            return obj

        summary_path = os.path.join(self.output_dir, 'analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(convert_numpy_types(self.results_summary), f, indent=2)

        return summary_path

    def run_complete_analysis(self):
        """Run complete analysis pipeline."""
        print("Starting comprehensive hyperparameter analysis...")
        print("="*60)

        try:
            # Load and clean data
            self.load_and_clean_data()

            # Perform statistical analysis
            self.statistical_analysis()

            # Find optimal configurations
            self.find_optimal_configurations()

            # Create visualizations
            self.create_performance_heatmaps()
            self.create_comparative_analysis()

            # Generate reports
            report_path = self.generate_comprehensive_report()
            summary_path = self.save_results_summary()

            print("\n" + "="*60)
            print("ANALYSIS COMPLETE!")
            print("="*60)
            print(f"Comprehensive report: {report_path}")
            print(f"Results summary: {summary_path}")
            print(f"Visualizations saved in: {self.output_dir}")

            return True

        except Exception as e:
            print(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description='Comprehensive hyperparameter optimization analysis')
    parser.add_argument('--summary_file', required=True,
                       help='Path to hyperparameter summary CSV file')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for analysis results')

    args = parser.parse_args()

    # Create analyzer and run complete analysis
    analyzer = HyperparameterAnalyzer(args.summary_file, args.output_dir)
    success = analyzer.run_complete_analysis()

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())