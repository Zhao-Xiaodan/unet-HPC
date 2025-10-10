#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Comprehensive_CNN_Architecture_Study
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# COMPREHENSIVE CNN ARCHITECTURE STUDY - ENHANCED PBS SCRIPT
# =======================================================================
# Combines ALL architectures from Skip Connections and UNet Parameter studies:
# - Baseline CNNs (Shallow/Deep, no skip connections)
# - ResNet variants (Shallow/Deep, with residual blocks)  
# - UNet variants (Multiple configurations, encoder-decoder)
# - DenseNet style (Dense connections)
#
# Enhanced Features:
# - Memory optimization and monitoring
# - Comprehensive analysis and visualization
# - JSON serialization fixes
# - Detailed progress tracking
# - Error recovery and logging
# =======================================================================

echo "======================================================================="
echo "COMPREHENSIVE CNN ARCHITECTURE STUDY - ENHANCED EXECUTION"
echo "======================================================================="
echo "Study: Complete comparison of ALL CNN architectures for density estimation"
echo "Enhanced: Memory optimization, comprehensive analysis, detailed visualization"
echo ""
echo "🚨 KEY ENHANCEMENTS FROM PREVIOUS STUDIES:"
echo "✅ Fixed JSON serialization issues (int64 -> int conversion)"
echo "✅ Optimized memory management with aggressive cleanup"
echo "✅ Added comprehensive plotting (Q-Q plots, density range analysis)"
echo "✅ Enhanced gradient flow tracking and visualization"
echo "✅ Statistical hypothesis testing framework"
echo "✅ Cross-architecture performance comparison"
echo "======================================================================="

# Job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $(free -h | grep Mem | awk '{print $2}'), CPUs: $(nproc)"
echo ""

# =======================================================================
# ENHANCED ENVIRONMENT SETUP
# =======================================================================

echo "=== COMPREHENSIVE STUDY CONFIGURATION ==="
echo "Dataset: ./dataset_preprocessed"
echo "Data Percentage: 50%"
echo "Learning Rate: 3e-4"
echo "Max Epochs: 50 (with early stopping)"
echo "Patience: 15"
echo "Base Workers: 8 (adaptive per architecture)"
echo "Base Batch Size: 64 (adaptive per architecture)"
echo "Mixed Precision: Enabled"
echo "Gradient Tracking: Enabled with detailed analysis"
echo "Memory Management: Enhanced with architecture-specific optimization"
echo "Conservative Mode: Enabled for HPC stability"
echo "Cleanup Frequency: Every 5 batches"
echo "Dilution Factors: 80x 160x 320x 640x 1280x 2560x 5120x 10240x"
echo "=============================================="
echo ""

# Memory optimization settings - Fixed for HPC CUDA allocator compatibility
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# Load required modules
module load singularity

# Define singularity container
image=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif

if [ ! -f "$image" ]; then
    echo "❌ ERROR: Container not found at $image"
    echo "Please check container path"
    exit 1
fi

echo "✅ Container found: $image"

# =======================================================================
# ENHANCED EXECUTION MONITORING
# =======================================================================

echo "=== GPU MEMORY STATUS BEFORE COMPREHENSIVE STUDY ==="
singularity exec --nv "$image" python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU Memory: {:.1f} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1024**3))
    print('GPU Memory - Total, Used, Free (MB):')
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'], capture_output=True, text=True)
        print(result.stdout)
    except:
        print('nvidia-smi not available')
else:
    print('CUDA not available')
"
echo "================================================="
echo ""

# =======================================================================
# COMPREHENSIVE ARCHITECTURE STUDY EXECUTION
# =======================================================================

echo "🚀 Starting Comprehensive CNN Architecture Study..."
echo "Testing ALL architectures: Baseline + ResNet + UNet + DenseNet variants"
echo "Enhanced with detailed analysis and visualization"
echo "=================================================="
echo ""

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="comprehensive_architecture_study_${TIMESTAMP}"

echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Execute the comprehensive study with enhanced parameters
cd /home/svu/phyzxi/scratch/densityCNN-HPC
singularity exec --nv "$image" python3 train_comprehensive_architecture_study.py \
    --input_dir ./dataset_preprocessed \
    --output_dir ./$OUTPUT_DIR \
    --epochs 50 \
    --patience 15 \
    --learning_rate 3e-4 \
    --base_batch_size 64 \
    --base_num_workers 8 \
    --data_percentage 50 \
    --mixed_precision \
    --track_gradients \
    --run_baselines \
    --run_resnets \
    --run_unets \
    --run_densenet \
    --memory_efficient \
    --conservative_mode \
    --cleanup_frequency 5 \
    --seed 42 2>&1 | tee comprehensive_study_console_${TIMESTAMP}.log

# Capture exit code
EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "COMPREHENSIVE CNN ARCHITECTURE STUDY COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✅ SUCCESS"
else
    echo "Exit code: $EXIT_CODE ❌ ERROR"
fi
echo "Study output directory: ./$OUTPUT_DIR"
echo ""

# =======================================================================
# POST-EXECUTION ANALYSIS AND REPORTING
# =======================================================================

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Comprehensive Architecture Study completed successfully!"
    echo ""
    
    echo "📊 Generated Analysis Files:"
    echo "📈 Performance Overview:"
    if [ -f "./$OUTPUT_DIR/comprehensive_architecture_comparison.csv" ]; then
        echo "✅ comprehensive_architecture_comparison.csv - Complete performance comparison"
        echo "   Top 3 performers by R² score:"
        singularity exec --nv "$image" python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('./$OUTPUT_DIR/comprehensive_architecture_comparison.csv')
    top3 = df.nlargest(3, 'R2_Score')[['Model', 'Architecture_Type', 'R2_Score', 'Parameters', 'Training_Time_Min']]
    print(top3.to_string(index=False))
except Exception as e:
    print('Error reading results:', e)
"
    else
        echo "❌ comprehensive_architecture_comparison.csv not found"
    fi
    echo ""
    
    echo "🔬 Complete Study Results:"
    if [ -f "./$OUTPUT_DIR/complete_comprehensive_study.json" ]; then
        echo "✅ complete_comprehensive_study.json - Full experimental results"
        echo "   Study summary:"
        singularity exec --nv "$image" python3 -c "
import json
import sys
try:
    with open('./$OUTPUT_DIR/complete_comprehensive_study.json', 'r') as f:
        data = json.load(f)
    info = data['study_info']
    print(f\"   Total time: {info['total_time_minutes']:.2f} minutes\")
    print(f\"   Successful experiments: {info['successful_experiments']}\")
    print(f\"   Total architectures tested: {info['total_architectures_tested']}\")
except Exception as e:
    print('Error reading study info:', e)
"
    else
        echo "❌ complete_comprehensive_study.json not found"
    fi
    echo ""
    
    echo "📂 Individual Experiment Results:"
    if ls ./$OUTPUT_DIR/experiment_*_results.json 1> /dev/null 2>&1; then
        echo "✅ Individual experiment files found:"
        ls -la ./$OUTPUT_DIR/experiment_*_results.json | wc -l | xargs echo "   Count:"
    else
        echo "❌ No individual experiment results found"
    fi
    echo ""
    
    echo "🏆 Best Model Checkpoints:"
    if ls ./$OUTPUT_DIR/best_model_*.pth 1> /dev/null 2>&1; then
        echo "✅ Model checkpoints found:"
        ls -la ./$OUTPUT_DIR/best_model_*.pth | wc -l | xargs echo "   Count:"
    else
        echo "❌ No model checkpoints found"
    fi
    echo ""
    
    echo "📈 Visualization Files:"
    if ls ./$OUTPUT_DIR/*.png 1> /dev/null 2>&1; then
        echo "✅ Analysis plots generated:"
        echo "   Training analysis plots:"
        ls ./$OUTPUT_DIR/training_analysis_*.png 2>/dev/null | wc -l | xargs echo "     Count:"
        echo "   Evaluation plots (with Q-Q plots, density range analysis):"
        ls ./$OUTPUT_DIR/evaluation_*.png 2>/dev/null | wc -l | xargs echo "     Count:"
        echo "   Gradient analysis plots:"
        ls ./$OUTPUT_DIR/gradient_analysis_*.png 2>/dev/null | wc -l | xargs echo "     Count:"
    else
        echo "❌ No visualization files found"
    fi
    echo ""
    
else
    echo "❌ Comprehensive Architecture Study failed!"
    echo "Check the console log for detailed error information:"
    echo "   comprehensive_study_console_${TIMESTAMP}.log"
    echo ""
    
    echo "🔍 Error Analysis:"
    if [ -f "comprehensive_study_console_${TIMESTAMP}.log" ]; then
        echo "Last 20 lines of console log:"
        tail -20 "comprehensive_study_console_${TIMESTAMP}.log"
    fi
    echo ""
    
    echo "📂 Available Error Files:"
    if ls ./$OUTPUT_DIR/*_ERROR.json 1> /dev/null 2>&1; then
        echo "✅ Error files found for debugging:"
        ls -la ./$OUTPUT_DIR/*_ERROR.json
    else
        echo "❌ No error files generated"
    fi
fi

echo ""
echo "=== COMPREHENSIVE STUDY RESULTS SUMMARY ==="
echo "=============================================="

# Architecture-specific performance summary
if [ -f "./$OUTPUT_DIR/comprehensive_architecture_comparison.csv" ] && [ $EXIT_CODE -eq 0 ]; then
    echo "🏆 PERFORMANCE RESULTS BY ARCHITECTURE TYPE:"
    echo "============================================"
    
    singularity exec --nv "$image" python3 -c "
import pandas as pd
import numpy as np
try:
    df = pd.read_csv('./$OUTPUT_DIR/comprehensive_architecture_comparison.csv')
    
    print('🥇 BASELINE ARCHITECTURES (No Skip Connections):')
    baseline = df[df['Architecture_Type'] == 'Baseline'].sort_values('R2_Score', ascending=False)
    for _, row in baseline.iterrows():
        print(f'   {row[\"Model\"]:20} | R² = {row[\"R2_Score\"]:.4f} | Params: {row[\"Parameters\"]:7,} | Time: {row[\"Training_Time_Min\"]:5.1f}min')
    
    print()
    print('🥈 RESNET ARCHITECTURES (With Skip Connections):')
    resnet = df[df['Architecture_Type'] == 'ResNet'].sort_values('R2_Score', ascending=False)
    for _, row in resnet.iterrows():
        print(f'   {row[\"Model\"]:20} | R² = {row[\"R2_Score\"]:.4f} | Params: {row[\"Parameters\"]:7,} | Time: {row[\"Training_Time_Min\"]:5.1f}min')
    
    print()
    print('🥉 UNET ARCHITECTURES (Encoder-Decoder + Skip):')
    unet = df[df['Architecture_Type'] == 'UNet'].sort_values('R2_Score', ascending=False)
    for _, row in unet.iterrows():
        print(f'   {row[\"Model\"]:20} | R² = {row[\"R2_Score\"]:.4f} | Params: {row[\"Parameters\"]:7,} | Time: {row[\"Training_Time_Min\"]:5.1f}min')
    
    print()
    print('🏅 DENSENET ARCHITECTURES (Dense Connections):')
    densenet = df[df['Architecture_Type'] == 'DenseNet'].sort_values('R2_Score', ascending=False)
    for _, row in densenet.iterrows():
        print(f'   {row[\"Model\"]:20} | R² = {row[\"R2_Score\"]:.4f} | Params: {row[\"Parameters\"]:7,} | Time: {row[\"Training_Time_Min\"]:5.1f}min')
    
    print()
    print('📊 ARCHITECTURE TYPE AVERAGES:')
    print('==============================')
    for arch_type in df['Architecture_Type'].unique():
        subset = df[df['Architecture_Type'] == arch_type]
        avg_r2 = subset['R2_Score'].mean()
        avg_params = subset['Parameters'].mean()
        avg_time = subset['Training_Time_Min'].mean()
        count = len(subset)
        print(f'   {arch_type:12} | Avg R² = {avg_r2:.4f} | Avg Params: {avg_params:7,.0f} | Avg Time: {avg_time:5.1f}min | Count: {count}')
    
except Exception as e:
    print('Error analyzing results:', e)
"
    
    echo ""
    echo "🎯 KEY INSIGHTS FROM COMPREHENSIVE STUDY:"
    echo "========================================"
    singularity exec --nv "$image" python3 -c "
import pandas as pd
import numpy as np
try:
    df = pd.read_csv('./$OUTPUT_DIR/comprehensive_architecture_comparison.csv')
    
    # Best overall
    best = df.loc[df['R2_Score'].idxmax()]
    print(f'🏆 Best Overall: {best[\"Model\"]} (R² = {best[\"R2_Score\"]:.4f})')
    
    # Best by architecture type
    for arch_type in df['Architecture_Type'].unique():
        subset = df[df['Architecture_Type'] == arch_type]
        if len(subset) > 0:
            best_in_type = subset.loc[subset['R2_Score'].idxmax()]
            print(f'🏆 Best {arch_type}: {best_in_type[\"Model\"]} (R² = {best_in_type[\"R2_Score\"]:.4f})')
    
    # Parameter efficiency
    df['param_efficiency'] = df['R2_Score'] / (df['Parameters'] / 1e6)
    most_efficient = df.loc[df['param_efficiency'].idxmax()]
    print(f'⚡ Most Parameter Efficient: {most_efficient[\"Model\"]} (R²/M params = {most_efficient[\"param_efficiency\"]:.2f})')
    
    # Time efficiency
    df['time_efficiency'] = df['R2_Score'] / df['Training_Time_Min']
    fastest = df.loc[df['time_efficiency'].idxmax()]
    print(f'🚀 Fastest Training: {fastest[\"Model\"]} (R²/min = {fastest[\"time_efficiency\"]:.3f})')
    
    # Skip connection analysis
    skip_true = df[df['Has_Skip_Connections'] == True]['R2_Score'].mean()
    skip_false = df[df['Has_Skip_Connections'] == False]['R2_Score'].mean()
    print(f'📊 Skip Connections Impact:')
    print(f'   With Skip Connections: Avg R² = {skip_true:.4f}')
    print(f'   Without Skip Connections: Avg R² = {skip_false:.4f}')
    print(f'   Difference: {skip_true - skip_false:+.4f}')
    
except Exception as e:
    print('Error generating insights:', e)
"
else
    echo "❌ Unable to generate performance summary - results file not available"
fi

echo ""
echo "=== FINAL STUDY STATUS ==="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ COMPREHENSIVE CNN ARCHITECTURE STUDY SUCCESSFUL!"
    echo "📁 All results saved in: ./$OUTPUT_DIR"
    echo "📊 Comprehensive comparison of ALL CNN architectures completed"
    echo "🎯 Enhanced with detailed analysis, Q-Q plots, and statistical testing"
    echo ""
    echo "🔍 Next Steps:"
    echo "1. Review comprehensive_architecture_comparison.csv for performance rankings"
    echo "2. Examine individual evaluation plots for detailed model analysis"
    echo "3. Check statistical_analysis.json for hypothesis testing results"
    echo "4. Use best performing models for production deployment"
else
    echo "❌ COMPREHENSIVE CNN ARCHITECTURE STUDY FAILED"
    echo "📋 Check error logs and debug information above"
    echo "🔧 Common issues: Memory allocation, path problems, container issues"
fi

echo ""
echo "======================================================================="
echo "COMPREHENSIVE CNN ARCHITECTURE STUDY - EXECUTION COMPLETE"
echo "Study: Complete comparison of Baseline + ResNet + UNet + DenseNet"
echo "Enhanced: Memory optimization, comprehensive analysis, detailed plots"
echo "======================================================================="