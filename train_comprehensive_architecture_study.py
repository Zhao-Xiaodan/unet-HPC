#!/usr/bin/env python
"""
Comprehensive CNN Architecture Study: Complete Comparison
=========================================================
Combines ALL architectures from both Skip Connections and UNet Parameter studies:
- Baseline CNNs (Shallow/Deep, no skip connections)
- ResNet variants (Shallow/Deep, with residual blocks)
- UNet variants (Multiple configurations, encoder-decoder)
- DenseNet style (Dense connections)

Enhanced with comprehensive analysis and visualization:
- Detailed training tracking with gradient flow analysis
- Performance by density range evaluation
- Q-Q plot normality checks for residuals
- Statistical hypothesis testing
- Memory usage optimization
- JSON serialization fixes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # HPC compatibility - no display
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import json
import argparse
import gc
import psutil
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set conservative memory optimization settings
# HPC/Container optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async for performance
torch.backends.cudnn.benchmark = True

# Verify critical HPC setup
print("🔍 HPC Environment Check:")
print(f"   Python: {os.sys.version}")
print(f"   PyTorch: {torch.__version__}")
print(f"   Matplotlib backend: {matplotlib.get_backend()}")
print(f"   CUDA available: {torch.cuda.is_available()}")

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

parser = argparse.ArgumentParser(description='Comprehensive CNN Architecture Study')
parser.add_argument('--input_dir', type=str, default='dataset_preprocessed',
                    help='Input directory containing images/ and density.csv')
parser.add_argument('--output_dir', type=str, default='comprehensive_architecture_study',
                    help='Directory to save results')
parser.add_argument('--epochs', type=int, default=50,
                    help='Maximum number of epochs to train')
parser.add_argument('--patience', type=int, default=15,
                    help='Early stopping patience')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='Learning rate for optimizer')
parser.add_argument('--base_batch_size', type=int, default=64,
                    help='Base batch size (will be adapted per architecture)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--base_num_workers', type=int, default=8,
                    help='Base number of data loading workers (will be adapted)')
parser.add_argument('--data_percentage', type=int, default=50,
                    help='Percentage of data to use')
parser.add_argument('--dilution_factors', nargs='+', type=str,
                    default=['80x', '160x', '320x', '640x', '1280x', '2560x', '5120x', '10240x'],
                    help='Dilution factors to include')
parser.add_argument('--use_all_dilutions', action='store_true',
                    help='Use all available dilution factors')
parser.add_argument('--mixed_precision', action='store_true', default=True,
                    help='Use mixed precision training')
parser.add_argument('--track_gradients', action='store_true', default=True,
                    help='Track gradient norms during training')
parser.add_argument('--run_baselines', action='store_true', default=True,
                    help='Run baseline CNN architectures')
parser.add_argument('--run_resnets', action='store_true', default=True,
                    help='Run ResNet architectures')
parser.add_argument('--run_unets', action='store_true', default=True,
                    help='Run UNet architectures')
parser.add_argument('--run_densenet', action='store_true', default=True,
                    help='Run DenseNet style architecture')
parser.add_argument('--memory_efficient', action='store_true', default=True,
                    help='Use memory efficient training')
parser.add_argument('--conservative_mode', action='store_true', default=True,
                    help='Use conservative settings for better reliability')
parser.add_argument('--cleanup_frequency', type=int, default=5,
                    help='Memory cleanup frequency (every N batches)')

# NOTE: Arguments will be parsed in main() function to avoid module-level execution

def parse_arguments():
    """Parse command line arguments - called from main() to avoid module-level execution"""
    parsed_args = parser.parse_args()
    
    # Verify all required arguments are available
    required_args = ['base_batch_size', 'base_num_workers', 'cleanup_frequency', 'conservative_mode', 'memory_efficient']
    for arg in required_args:
        if not hasattr(parsed_args, arg):
            print(f"❌ Missing required argument: {arg}")
            print("   This indicates the patch script didn't apply correctly.")
            exit(1)
    
    return parsed_args

# Seeds and configuration will be set in main() function after argument parsing
torch.backends.cudnn.benchmark = True

# Module-level prints and args references moved to main() to avoid import conflicts


# ============================================================================
# ADAPTIVE RESOURCE MANAGEMENT
# ============================================================================

def get_architecture_config(model, args):
    """Get architecture-specific configuration for optimal performance"""
    arch_type = getattr(model, 'architecture_type', 'Unknown')
    param_count = sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else 0
    
    # Base configurations
    config = {
        'batch_size': args.base_batch_size,
        'num_workers': args.base_num_workers,
        'memory_cleanup_frequency': args.cleanup_frequency,
        'gradient_accumulation': 1,
        'use_checkpointing': False
    }
    
    # Architecture-specific optimizations
    if arch_type == 'UNet':
        # UNet models are memory intensive
        config['batch_size'] = max(16, args.base_batch_size // 2)  # Smaller batches
        config['num_workers'] = max(4, args.base_num_workers // 2)
        config['memory_cleanup_frequency'] = 3  # More frequent cleanup
        config['use_checkpointing'] = True
        config['gradient_accumulation'] = 2  # Simulate larger batch
        
    elif arch_type == 'ResNet' and param_count > 2000000:
        # Large ResNet models
        config['batch_size'] = max(32, args.base_batch_size * 3 // 4)
        config['memory_cleanup_frequency'] = 5
        
    elif arch_type == 'DenseNet':
        # DenseNet can be memory intensive due to concatenations
        config['batch_size'] = max(24, args.base_batch_size // 2)
        config['memory_cleanup_frequency'] = 4
        config['use_checkpointing'] = True
        
    elif arch_type == 'Baseline' and param_count < 1000000:
        # Small baseline models can use larger batches
        config['batch_size'] = min(128, args.base_batch_size * 2)
        config['num_workers'] = min(12, args.base_num_workers * 3 // 2)
        config['memory_cleanup_frequency'] = 10
    
    # Conservative mode overrides
    if args.conservative_mode:
        config['batch_size'] = min(config['batch_size'], 32)
        config['num_workers'] = min(config['num_workers'], 6)
        config['memory_cleanup_frequency'] = max(3, config['memory_cleanup_frequency'] // 2)
    
    print(f"📐 Architecture config for {getattr(model, 'name', 'Unknown')}: "
          f"batch={config['batch_size']}, workers={config['num_workers']}, "
          f"cleanup_freq={config['memory_cleanup_frequency']}")
    
    return config

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def fix_json_serialization(obj):
    """Fix JSON serialization for numpy types"""
    if isinstance(obj, dict):
        return {k: fix_json_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_json_serialization(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif np.isnan(obj) if isinstance(obj, (int, float, np.number)) else False:
        return None
    else:
        return obj

def enhanced_memory_cleanup(aggressive=False):
    """Enhanced memory cleanup with multiple strategies - HPC compatible"""
    if aggressive:
        print("🧹 Performing aggressive memory cleanup...")
    
    # Python garbage collection
    for _ in range(3 if aggressive else 1):
        gc.collect()
    
    # PyTorch cleanup with HPC-compatible settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if aggressive:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            # Clear PyTorch caching allocator
            torch.cuda.empty_cache()
    
    # Memory status
    if torch.cuda.is_available() and aggressive:
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"💾 GPU Memory - Allocated: {memory_allocated:.3f} GB, Reserved: {memory_reserved:.3f} GB")

def aggressive_memory_cleanup():
    """Backward compatibility alias for enhanced_memory_cleanup"""
    enhanced_memory_cleanup(aggressive=True)

# Backward compatibility alias removed - using single enhanced_memory_cleanup function

def create_robust_scheduler(optimizer, scheduler_type='cosine_warm'):
    """Create learning rate scheduler with fallback options"""
    try:
        if scheduler_type == 'cosine_warm':
            T_0 = int(10)  # Fixed: ensure integer
            T_mult = int(2)  # Fixed: ensure integer
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
            print(f"✅ Created CosineAnnealingWarmRestarts scheduler (T_0={T_0}, T_mult={T_mult})")
            return scheduler
    except Exception as e:
        print(f"❌ Scheduler creation failed: {str(e)}")
        print("🔄 Falling back to StepLR scheduler")
        try:
            fallback_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
            print("✅ Created fallback StepLR scheduler")
            return fallback_scheduler
        except Exception as fallback_error:
            print(f"❌ Fallback scheduler also failed: {str(fallback_error)}")
            return None

# ============================================================================
# BASELINE CNN ARCHITECTURES (From Skip Connections Study)
# ============================================================================

class BaselineShallowCNN(nn.Module):
    """Baseline 4-layer CNN without skip connections"""
    def __init__(self, base_filters=64):
        super(BaselineShallowCNN, self).__init__()
        self.name = "Baseline_Shallow"
        self.depth = 4
        self.has_skip_connections = False
        self.architecture_type = "Baseline"

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, base_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Linear(base_filters*4 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class BaselineDeepCNN(nn.Module):
    """Deep 12-layer CNN without skip connections"""
    def __init__(self, base_filters=32):
        super(BaselineDeepCNN, self).__init__()
        self.name = "Baseline_Deep"
        self.depth = 12
        self.has_skip_connections = False
        self.architecture_type = "Baseline"

        self.layers = nn.ModuleList()
        in_channels = 1
        current_filters = base_filters

        for i in range(12):
            if i in [0, 3, 6, 9]:
                if i > 0:
                    current_filters *= 2
                current_filters = min(current_filters, 512)

            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, current_filters, 3, padding=1, bias=False),
                nn.BatchNorm2d(current_filters),
                nn.ReLU(inplace=True)
            ))

            if i in [2, 5, 8, 11]:
                self.layers.append(nn.MaxPool2d(2, 2))

            in_channels = current_filters

        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(current_filters * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ============================================================================
# RESNET ARCHITECTURES (From Skip Connections Study)
# ============================================================================

class ResidualBlock(nn.Module):
    """Basic residual block for ResNet implementations"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNetShallowCNN(nn.Module):
    """4-layer ResNet-style CNN with residual blocks"""
    def __init__(self, base_filters=64):
        super(ResNetShallowCNN, self).__init__()
        self.name = "ResNet_Shallow"
        self.depth = 4
        self.has_skip_connections = True
        self.architecture_type = "ResNet"

        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, base_filters, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = ResidualBlock(base_filters, base_filters)
        self.layer2 = ResidualBlock(base_filters, base_filters*2, stride=2)
        self.layer3 = ResidualBlock(base_filters*2, base_filters*4, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(base_filters*4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class ResNetDeepCNN(nn.Module):
    """Deep 12-layer ResNet with residual blocks"""
    def __init__(self, base_filters=32):
        super(ResNetDeepCNN, self).__init__()
        self.name = "ResNet_Deep"
        self.depth = 12
        self.has_skip_connections = True
        self.architecture_type = "ResNet"

        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, base_filters, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(base_filters, base_filters, 2, stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters*2, 2, stride=2)
        self.layer3 = self._make_layer(base_filters*2, base_filters*4, 2, stride=2)
        self.layer4 = self._make_layer(base_filters*4, base_filters*8, 2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(base_filters*8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ============================================================================
# UNET ARCHITECTURES (From UNet Parameter Study)
# ============================================================================

class MemoryEfficientConvBlock(nn.Module):
    """Memory-efficient conv block with optional checkpointing"""
    def __init__(self, in_channels, out_channels, use_checkpoint=True):
        super(MemoryEfficientConvBlock, self).__init__()
        self.use_checkpoint = use_checkpoint
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self.conv, x, use_reentrant=False)
        else:
            return self.conv(x)

class ConservativeSkipConnection(nn.Module):
    """Conservative skip connection with channel reduction"""
    def __init__(self, skip_channels, up_channels, out_channels):
        super(ConservativeSkipConnection, self).__init__()
        
        reduced_skip_channels = min(skip_channels // 2, up_channels // 2, out_channels // 2)
        reduced_skip_channels = max(reduced_skip_channels, 8)
        
        self.skip_reducer = nn.Sequential(
            nn.Conv2d(skip_channels, reduced_skip_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_skip_channels)
        ) if skip_channels != reduced_skip_channels else nn.Identity()
        
        self.conv = MemoryEfficientConvBlock(
            up_channels + reduced_skip_channels, 
            out_channels,
            use_checkpoint=True
        )
    
    def forward(self, up_x, skip_x):
        skip_x = self.skip_reducer(skip_x)
        combined = torch.cat([up_x, skip_x], dim=1)
        return self.conv(combined)

class FullConcatSkipConnection(nn.Module):
    """Full concatenation skip connection"""
    def __init__(self, skip_channels, up_channels, out_channels):
        super(FullConcatSkipConnection, self).__init__()
        
        concat_channels = up_channels + skip_channels
        self.conv = MemoryEfficientConvBlock(
            concat_channels,
            out_channels,
            use_checkpoint=True
        )
    
    def forward(self, up_x, skip_x):
        combined = torch.cat([up_x, skip_x], dim=1)
        return self.conv(combined)

class UNetCNN(nn.Module):
    """UNet architecture with configurable skip connections"""
    def __init__(self, base_filters=32, skip_connection_type='channel_reduced', max_filters=128):
        super(UNetCNN, self).__init__()
        self.base_filters = base_filters
        self.skip_connection_type = skip_connection_type
        self.max_filters = max_filters
        self.name = f"UNet_{skip_connection_type}_{base_filters}filters"
        self.depth = 4
        self.has_skip_connections = True
        self.architecture_type = "UNet"
        
        # Conservative filter progression
        f1 = min(base_filters, max_filters)
        f2 = min(base_filters * 2, max_filters)
        f3 = min(base_filters * 3, max_filters)
        f4 = min(base_filters * 4, max_filters)
        
        # Encoder
        self.enc1 = MemoryEfficientConvBlock(1, f1, use_checkpoint=False)
        self.enc2 = MemoryEfficientConvBlock(f1, f2, use_checkpoint=True)
        self.enc3 = MemoryEfficientConvBlock(f2, f3, use_checkpoint=True)
        
        # Bottleneck
        self.bottleneck = MemoryEfficientConvBlock(f3, f4, use_checkpoint=True)
        
        # Decoder
        if skip_connection_type == 'channel_reduced':
            self.dec3 = ConservativeSkipConnection(f3, f4, f3)
            self.dec2 = ConservativeSkipConnection(f2, f3, f2)
            self.dec1 = ConservativeSkipConnection(f1, f2, f1)
        else:  # 'full_concat'
            self.dec3 = FullConcatSkipConnection(f3, f4, f3)
            self.dec2 = FullConcatSkipConnection(f2, f3, f2)
            self.dec1 = FullConcatSkipConnection(f1, f2, f1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        classifier_input_size = f1 * 16
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(classifier_input_size, min(128, f1 * 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(min(128, f1 * 2), min(32, f1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(min(32, f1), 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        x1 = self.pool(e1)
        
        e2 = self.enc2(x1)
        x2 = self.pool(e2)
        
        e3 = self.enc3(x2)
        x3 = self.pool(e3)
        
        # Bottleneck
        bottleneck = self.bottleneck(x3)
        
        # Decoder path
        up3 = self.upsample(bottleneck)
        d3 = self.dec3(up3, e3)
        
        up2 = self.upsample(d3)
        d2 = self.dec2(up2, e2)
        
        up1 = self.upsample(d2)
        d1 = self.dec1(up1, e1)
        
        # Final classification
        features = self.adaptive_pool(d1)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

# ============================================================================
# DENSENET ARCHITECTURE (From Skip Connections Study)
# ============================================================================

class DenseBlock(nn.Module):
    """Dense block with growth connections"""
    def __init__(self, in_channels, growth_rate=32):
        super(DenseBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate*4, 1, bias=False)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(growth_rate*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate*4, growth_rate, 3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return torch.cat([x, out], dim=1)

class DenseNetStyleCNN(nn.Module):
    """DenseNet-style CNN with dense connections"""
    def __init__(self, base_filters=64, growth_rate=32):
        super(DenseNetStyleCNN, self).__init__()
        self.name = "DenseNet_Style"
        self.depth = 4
        self.has_skip_connections = True
        self.architecture_type = "DenseNet"

        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, base_filters, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        num_features = base_filters
        self.dense1 = DenseBlock(num_features, growth_rate)
        num_features += growth_rate

        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features//2, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )
        num_features = num_features//2

        self.dense2 = DenseBlock(num_features, growth_rate)
        num_features += growth_rate

        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features//2, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )
        num_features = num_features//2

        self.dense3 = DenseBlock(num_features, growth_rate)
        num_features += growth_rate

        self.final_bn = nn.BatchNorm2d(num_features)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.final_bn(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ============================================================================
# DATASET AND PREPROCESSING
# ============================================================================

class MicrobeadDataset(Dataset):
    def __init__(self, image_dir, density_csv, transform=None, data_percentage=100,
                 dilution_factors=None, use_all_dilutions=False):
        self.image_dir = image_dir
        self.transform = transform

        self.df = pd.read_csv(density_csv)

        if len(self.df.columns) == 1:
            self.df = self.df.iloc[:, 0].str.split(expand=True)
            self.df.columns = ['filename', 'density']
        elif len(self.df.columns) == 2:
            self.df.columns = ['filename', 'density']

        self.df['density'] = self.df['density'].astype(float)

        if not use_all_dilutions and dilution_factors:
            pattern = '|'.join([f'^{factor}_' for factor in dilution_factors])
            mask = self.df['filename'].str.contains(pattern, case=False, na=False)
            self.df = self.df[mask].reset_index(drop=True)

        if data_percentage < 100:
            sample_size = int(len(self.df) * data_percentage / 100)
            self.df = self.df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)

        print(f"Dataset size: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename']
        if not img_name.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            img_name = img_name + '.png'

        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('L', (512, 512), 0)

        density = self.df.iloc[idx]['density']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(density, dtype=torch.float32)

def get_transforms():
    """Standard transforms for 512x512 images"""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.3),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    return train_transform, eval_transform

# ============================================================================
# GRADIENT TRACKING UTILITIES
# ============================================================================

class GradientTracker:
    """Track gradient norms during training"""
    def __init__(self):
        self.gradient_norms = []
        self.layer_gradient_norms = {}

    def track_gradients(self, model, epoch):
        total_norm = 0.0
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

                layer_type = name.split('.')[0]
                if layer_type not in layer_norms:
                    layer_norms[layer_type] = []
                layer_norms[layer_type].append(param_norm.item())

        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append((epoch, total_norm))

        for layer_type, norms in layer_norms.items():
            avg_norm = np.mean(norms)
            if layer_type not in self.layer_gradient_norms:
                self.layer_gradient_norms[layer_type] = []
            self.layer_gradient_norms[layer_type].append((epoch, avg_norm))

    def get_vanishing_gradient_score(self):
        """Calculate vanishing gradient severity score"""
        if not self.gradient_norms:
            return 0.0
        norms = [norm for _, norm in self.gradient_norms]
        vanishing_count = sum(1 for norm in norms if norm < 1e-6)
        return vanishing_count / len(norms)

# ============================================================================
# COMPREHENSIVE TRAINING FUNCTION
# ============================================================================

def train_model_comprehensive(model, train_loader, val_loader, config, device='cuda'):
    """Comprehensive training with detailed tracking"""
    print(f"Training {model.name} with comprehensive analysis...")
    
    enhanced_memory_cleanup(aggressive=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = create_robust_scheduler(optimizer)
    scaler = GradScaler() if config.get('mixed_precision', False) else None
    gradient_tracker = GradientTracker() if config.get('track_gradients', False) else None

    model.to(device)

    # Training tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'training_stability': [],
        'memory_usage': [],
        'memory_peak': []
    }

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve_epochs = 0
    epochs_completed = 0
    recent_losses = []
    stability_window = 5

    start_time = time.time()

    try:
        for epoch in range(config['num_epochs']):
            epochs_completed = epoch + 1
            epoch_start_time = time.time()

            # Reset memory stats
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch_idx, (images, densities) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                densities = densities.to(device, non_blocking=True)

                optimizer.zero_grad()

                try:
                    if config.get('mixed_precision', False) and scaler:
                        with autocast():
                            outputs = model(images).squeeze()
                            loss = criterion(outputs, densities)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(images).squeeze()
                        loss = criterion(outputs, densities)
                        loss.backward()
                        optimizer.step()

                    train_loss += loss.item()
                    num_batches += 1

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"❌ OOM during training at batch {batch_idx}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

                del images, densities, outputs, loss
                
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

            # Track gradients
            if gradient_tracker:
                gradient_tracker.track_gradients(model, epoch)

            avg_train_loss = train_loss / num_batches if num_batches > 0 else float('inf')
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            model.eval()
            val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for images, densities in val_loader:
                    images = images.to(device, non_blocking=True)
                    densities = densities.to(device, non_blocking=True)

                    try:
                        if config.get('mixed_precision', False):
                            with autocast():
                                outputs = model(images).squeeze()
                                loss = criterion(outputs, densities)
                        else:
                            outputs = model(images).squeeze()
                            loss = criterion(outputs, densities)

                        val_loss += loss.item()
                        num_val_batches += 1

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"❌ OOM during validation")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e

                    del images, densities, outputs, loss

            avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
            history['val_loss'].append(avg_val_loss)

            # Memory tracking
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**3
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                history['memory_usage'].append(current_memory)
                history['memory_peak'].append(peak_memory)

            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            # Training stability
            recent_losses.append(avg_val_loss)
            if len(recent_losses) > stability_window:
                recent_losses.pop(0)
            stability_score = np.std(recent_losses) if len(recent_losses) >= 3 else 0.0
            history['training_stability'].append(stability_score)

            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            epoch_time = time.time() - epoch_start_time
            memory_str = f"Mem: {history['memory_usage'][-1]:.3f}GB Peak: {history['memory_peak'][-1]:.3f}GB" if history['memory_usage'] else ""

            print(f'Epoch {epoch+1:3d}/{config["num_epochs"]:3d} | '
                  f'Train: {avg_train_loss:.2f} | Val: {avg_val_loss:.2f} | '
                  f'LR: {current_lr:.6f} | Stability: {stability_score:.4f} | '
                  f'{memory_str} | Time: {epoch_time:.1f}s')

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                no_improve_epochs = 0
                
                # Save best model
                model_path = os.path.join(config['output_dir'], f"best_model_{model.name}.pth")
                torch.save(model.state_dict(), model_path)
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

            torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise e

    total_time = time.time() - start_time
    training_minutes = total_time / 60

    # Restore best weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Generate comprehensive training analysis plots
    create_training_analysis_plots(model, history, gradient_tracker, config)

    training_stats = {
        'model_name': model.name,
        'architecture_info': {
            'architecture_type': getattr(model, 'architecture_type', 'Unknown'),
            'depth': model.depth,
            'has_skip_connections': model.has_skip_connections,
            'parameters': sum(p.numel() for p in model.parameters())
        },
        'training_performance': {
            'epochs_completed': epochs_completed,
            'best_val_loss': float(best_val_loss),
            'training_minutes': training_minutes,
            'final_train_loss': float(history['train_loss'][-1]) if history['train_loss'] else float('inf'),
            'convergence_epoch': epochs_completed - no_improve_epochs
        },
        'stability_metrics': {
            'avg_stability_score': float(np.mean(history['training_stability'])) if history['training_stability'] else 0.0,
            'final_stability_score': float(history['training_stability'][-1]) if history['training_stability'] else 0.0,
            'loss_variance': float(np.var(history['val_loss'])) if history['val_loss'] else 0.0
        },
        'gradient_analysis': {
            'vanishing_gradient_score': gradient_tracker.get_vanishing_gradient_score() if gradient_tracker else 0.0,
            'avg_gradient_norm': float(np.mean([norm for _, norm in gradient_tracker.gradient_norms])) if gradient_tracker and gradient_tracker.gradient_norms else 0.0,
            'gradient_stability': float(np.std([norm for _, norm in gradient_tracker.gradient_norms])) if gradient_tracker and gradient_tracker.gradient_norms else 0.0
        },
        'memory_metrics': {
            'max_memory_gb': float(max(history['memory_peak'])) if history['memory_peak'] else 0.0,
            'avg_memory_gb': float(np.mean(history['memory_usage'])) if history['memory_usage'] else 0.0
        },
        'history': fix_json_serialization({
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'learning_rates': history['learning_rates'],
            'training_stability': history['training_stability'],
            'memory_usage': history['memory_usage'],
            'memory_peak': history['memory_peak']
        })
    }

    return model, training_stats, gradient_tracker

# ============================================================================
# COMPREHENSIVE EVALUATION WITH DETAILED PLOTS
# ============================================================================

def evaluate_model_comprehensive(model, test_loader, config, device):
    """Comprehensive evaluation with detailed analysis plots"""
    model.eval()
    predictions = []
    actual_values = []
    inference_times = []

    print(f"Evaluating {model.name} with comprehensive analysis...")

    with torch.no_grad():
        for batch_idx, (images, densities) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            densities = densities.to(device, non_blocking=True)

            start_time = time.time()

            try:
                if config.get('mixed_precision', False):
                    with autocast():
                        outputs = model(images).squeeze()
                else:
                    outputs = model(images).squeeze()

                inference_time = (time.time() - start_time) * 1000
                inference_times.append(inference_time)

                predictions.extend(outputs.cpu().numpy())
                actual_values.extend(densities.cpu().numpy())

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ OOM during evaluation at batch {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            del images, densities, outputs

            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

    if not predictions or not actual_values:
        print("❌ No valid predictions obtained")
        return None

    predictions = np.array(predictions)
    actual_values = np.array(actual_values)

    # Calculate comprehensive metrics
    mse = np.mean((predictions - actual_values) ** 2)
    mae = np.mean(np.abs(predictions - actual_values))
    rmse = np.sqrt(mse)

    ss_res = np.sum((actual_values - predictions) ** 2)
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    mape = np.mean(np.abs((actual_values - predictions) / np.clip(actual_values, 1e-8, None))) * 100
    max_error = np.max(np.abs(actual_values - predictions))
    avg_inference_time = np.mean(inference_times)

    # Statistical tests
    residuals = actual_values - predictions
    _, normality_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)

    print(f"Evaluation Results for {model.name}:")
    print(f"  R² Score: {r2:.6f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Avg Inference: {avg_inference_time:.2f}ms/batch")
    print(f"  Residuals Normality p-value: {normality_p:.4f}")

    # Create comprehensive evaluation plots
    create_evaluation_plots(model, predictions, actual_values, residuals, normality_p, 
                          avg_inference_time, config)

    evaluation_metrics = {
        'model_name': model.name,
        'performance_metrics': {
            'r2_score': float(r2),
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'max_error': float(max_error)
        },
        'efficiency_metrics': {
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'avg_inference_time_ms': float(avg_inference_time),
            'parameter_efficiency': float(r2 / (sum(p.numel() for p in model.parameters()) / 1e6))
        },
        'statistical_analysis': {
            'residuals_normality_p': float(normality_p),
            'residuals_mean': float(np.mean(residuals)),
            'residuals_std': float(np.std(residuals))
        }
    }

    return evaluation_metrics

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_training_analysis_plots(model, history, gradient_tracker, config):
    """Create comprehensive training analysis plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Loss curves
    axes[0,0].plot(history['train_loss'], label='Training Loss', linewidth=2)
    axes[0,0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title(f'{model.name} - Training Progress')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[0,1].plot(history['learning_rates'], linewidth=2, color='orange')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Learning Rate')
    axes[0,1].set_title('Learning Rate Schedule')
    axes[0,1].grid(True, alpha=0.3)

    # Training stability
    axes[0,2].plot(history['training_stability'], linewidth=2, color='green')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('Loss Std (5-epoch window)')
    axes[0,2].set_title('Training Stability')
    axes[0,2].grid(True, alpha=0.3)

    # Memory usage
    if history['memory_usage']:
        axes[1,0].plot(history['memory_usage'], label='Current', linewidth=2, color='blue')
        axes[1,0].plot(history['memory_peak'], label='Peak', linewidth=2, color='red')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Memory Usage (GB)')
        axes[1,0].set_title('GPU Memory Usage')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

    # Gradient analysis
    if gradient_tracker and gradient_tracker.gradient_norms:
        epochs, norms = zip(*gradient_tracker.gradient_norms)
        axes[1,1].plot(epochs, norms, linewidth=2, color='red')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Gradient Norm')
        axes[1,1].set_title('Gradient Flow Analysis')
        axes[1,1].grid(True, alpha=0.3)

        vanishing_threshold = 1e-6
        vanishing_epochs = [e for e, n in gradient_tracker.gradient_norms if n < vanishing_threshold]
        if vanishing_epochs:
            axes[1,1].axhline(y=vanishing_threshold, color='red', linestyle='--',
                           label=f'Vanishing ({len(vanishing_epochs)} epochs)')
            axes[1,1].legend()

    # Model summary
    param_count = sum(p.numel() for p in model.parameters())
    training_time = len(history['train_loss']) * np.mean([15]) if history['train_loss'] else 0  # estimate
    
    axes[1,2].text(0.1, 0.9, f'Architecture: {model.name}', fontsize=14, weight='bold')
    axes[1,2].text(0.1, 0.8, f'Type: {getattr(model, "architecture_type", "Unknown")}', fontsize=12)
    axes[1,2].text(0.1, 0.7, f'Depth: {model.depth} layers', fontsize=12)
    axes[1,2].text(0.1, 0.6, f'Skip Connections: {model.has_skip_connections}', fontsize=12)
    axes[1,2].text(0.1, 0.5, f'Parameters: {param_count:,}', fontsize=12)
    axes[1,2].text(0.1, 0.4, f'Training Time: {training_time/60:.1f} min', fontsize=12)
    
    if history['val_loss']:
        best_val_loss = min(history['val_loss'])
        axes[1,2].text(0.1, 0.2, f'Best Val Loss: {best_val_loss:.4f}',
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    axes[1,2].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(config['output_dir'], f'training_analysis_{model.name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_evaluation_plots(model, predictions, actual_values, residuals, normality_p, 
                          avg_inference_time, config):
    """Create comprehensive evaluation plots with Q-Q plot and density range analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Predictions vs Actual
    min_val, max_val = min(min(actual_values), min(predictions)), max(max(actual_values), max(predictions))
    axes[0,0].scatter(actual_values, predictions, alpha=0.6, s=20)
    axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0,0].set_xlabel('Actual Density')
    axes[0,0].set_ylabel('Predicted Density')
    
    r2 = 1 - np.sum((actual_values - predictions)**2) / np.sum((actual_values - np.mean(actual_values))**2)
    axes[0,0].set_title(f'{model.name}: Predictions vs Actual\nR² = {r2:.4f}')
    axes[0,0].grid(True, alpha=0.3)

    # Residuals plot
    axes[0,1].scatter(actual_values, residuals, alpha=0.6, s=20)
    axes[0,1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0,1].set_xlabel('Actual Density')
    axes[0,1].set_ylabel('Residuals')
    mae = np.mean(np.abs(residuals))
    axes[0,1].set_title(f'Residual Analysis\nMAE = {mae:.4f}')
    axes[0,1].grid(True, alpha=0.3)

    # Q-Q plot for normality check
    stats.probplot(residuals, dist="norm", plot=axes[0,2])
    axes[0,2].set_title(f'Q-Q Plot (Normality Check)\np-value: {normality_p:.4f}')
    axes[0,2].grid(True, alpha=0.3)

    # Performance by density range - Dynamic range detection
    max_density = int(np.max(actual_values))
    min_density = int(np.min(actual_values))
    
    # Create dynamic ranges that cover the full data spectrum
    if max_density <= 1000:
        density_ranges = [(0, 50), (50, 150), (150, 300), (300, max_density)]
    else:
        # For high-density data, create more comprehensive ranges
        density_ranges = [
            (0, 50), (50, 150), (150, 300), (300, 600),
            (600, 1000), (1000, 2000), (2000, 3000), (3000, max_density)
        ]
        # Filter out empty ranges
        density_ranges = [(low, high) for low, high in density_ranges if high > low]
    
    range_performance = []

    for min_d, max_d in density_ranges:
        mask = (actual_values >= min_d) & (actual_values <= max_d)
        if np.sum(mask) > 10:
            range_r2 = 1 - np.sum((actual_values[mask] - predictions[mask])**2) / \
                           np.sum((actual_values[mask] - np.mean(actual_values[mask]))**2)
            range_performance.append((f"{min_d}-{max_d}", range_r2, np.sum(mask)))

    if range_performance:
        ranges, r2_scores, counts = zip(*range_performance)
        bars = axes[1,0].bar(ranges, r2_scores, color='skyblue', alpha=0.8)
        axes[1,0].set_xlabel('Density Range')
        axes[1,0].set_ylabel('R² Score')
        axes[1,0].set_title('Performance by Density Range')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'n={count}', ha='center', va='bottom', fontsize=9)

    # Error distribution
    axes[1,1].hist(residuals, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1,1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1,1].set_xlabel('Prediction Error')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title(f'Error Distribution\nStd: {np.std(residuals):.4f}')
    axes[1,1].grid(True, alpha=0.3)

    # Model architecture summary
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    axes[1,2].text(0.1, 0.9, f'Architecture: {model.name}', fontsize=14, weight='bold')
    axes[1,2].text(0.1, 0.8, f'Type: {getattr(model, "architecture_type", "Unknown")}', fontsize=12)
    axes[1,2].text(0.1, 0.7, f'Depth: {model.depth} layers', fontsize=12)
    axes[1,2].text(0.1, 0.6, f'Skip Connections: {model.has_skip_connections}', fontsize=12)
    axes[1,2].text(0.1, 0.5, f'Parameters: {param_count:,}', fontsize=12)
    axes[1,2].text(0.1, 0.4, f'Trainable: {trainable_params:,}', fontsize=12)
    axes[1,2].text(0.1, 0.3, f'Avg Inference: {avg_inference_time:.2f}ms', fontsize=12)
    axes[1,2].text(0.1, 0.1, f'Performance:\n  R²: {r2:.6f}\n  RMSE: {np.sqrt(np.mean(residuals**2)):.4f}',
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    axes[1,2].axis('off')

    plt.tight_layout()
    eval_plot_path = os.path.join(config['output_dir'], f'evaluation_{model.name}.png')
    plt.savefig(eval_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def perform_comprehensive_statistical_analysis(all_results, output_dir):
    """Perform comprehensive statistical analysis"""
    print("📊 Performing comprehensive statistical analysis...")
    
    # Extract data
    models = []
    r2_scores = []
    training_times = []
    parameter_counts = []
    gradient_scores = []
    stability_scores = []
    architecture_types = []
    
    for result in all_results:
        models.append(result['training']['model_name'])
        r2_scores.append(result['evaluation']['performance_metrics']['r2_score'])
        training_times.append(result['training']['training_performance']['training_minutes'])
        parameter_counts.append(result['evaluation']['efficiency_metrics']['parameters'])
        gradient_scores.append(result['training']['gradient_analysis']['vanishing_gradient_score'])
        stability_scores.append(result['training']['stability_metrics']['avg_stability_score'])
        architecture_types.append(result['training']['architecture_info']['architecture_type'])
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': models,
        'Architecture_Type': architecture_types,
        'R2_Score': r2_scores,
        'Training_Time_Min': training_times,
        'Parameters': parameter_counts,
        'Vanishing_Gradient_Score': gradient_scores,
        'Training_Stability': stability_scores,
        'Has_Skip_Connections': [result['training']['architecture_info']['has_skip_connections'] for result in all_results],
        'Depth': [result['training']['architecture_info']['depth'] for result in all_results]
    })
    
    # Statistical tests by architecture type
    statistical_results = {}
    
    # Group by architecture type for analysis
    baseline_r2 = comparison_df[comparison_df['Architecture_Type'] == 'Baseline']['R2_Score'].values
    resnet_r2 = comparison_df[comparison_df['Architecture_Type'] == 'ResNet']['R2_Score'].values
    unet_r2 = comparison_df[comparison_df['Architecture_Type'] == 'UNet']['R2_Score'].values
    
    # Perform pairwise comparisons
    if len(baseline_r2) > 0 and len(resnet_r2) > 0:
        t_stat, p_val = stats.ttest_ind(baseline_r2, resnet_r2)
        statistical_results['baseline_vs_resnet'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'baseline_mean': float(np.mean(baseline_r2)),
            'resnet_mean': float(np.mean(resnet_r2)),
            'effect_size': float(np.mean(baseline_r2) - np.mean(resnet_r2))
        }
    
    if len(baseline_r2) > 0 and len(unet_r2) > 0:
        t_stat, p_val = stats.ttest_ind(baseline_r2, unet_r2)
        statistical_results['baseline_vs_unet'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'baseline_mean': float(np.mean(baseline_r2)),
            'unet_mean': float(np.mean(unet_r2)),
            'effect_size': float(np.mean(baseline_r2) - np.mean(unet_r2))
        }
    
    # Save results
    comparison_df.to_csv(os.path.join(output_dir, 'comprehensive_architecture_comparison.csv'), index=False)
    
    with open(os.path.join(output_dir, 'statistical_analysis.json'), 'w') as f:
        json.dump(fix_json_serialization(statistical_results), f, indent=4)
    
    return statistical_results, comparison_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================


def verify_hpc_environment():
    """Verify HPC/container environment compatibility"""
    print("🔍 Verifying HPC environment...")
    
    # Check essential dependencies
    try:
        import torch
        import torchvision  
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import scipy
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ CUDA not available - will use CPU")
    
    # Check matplotlib backend
    print(f"✅ Matplotlib backend: {matplotlib.get_backend()}")
    
    return True

def main():
    # Parse arguments first (moved from module level to avoid import conflicts)
    args = parse_arguments()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Verify HPC environment first
    if not verify_hpc_environment():
        print("❌ Environment verification failed")
        exit(1)
        
    print(f"🔬 COMPREHENSIVE CNN ARCHITECTURE STUDY")
    print("=" * 80)
    print(f"🎯 Objective: Complete comparison of ALL CNN architectures")
    print(f"📊 Dataset: {args.data_percentage}% of {args.input_dir}")
    print(f"🧪 Architectures: Baseline + ResNet + UNet + DenseNet variants")
    print("=" * 80)
    
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, args.input_dir)
    image_dir = os.path.join(input_path, 'images')
    density_file = os.path.join(input_path, 'density.csv')
    
    # Validate paths
    for path, name in [(input_path, 'Input directory'),
                       (image_dir, 'Images directory'),
                       (density_file, 'Density file')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} does not exist: {path}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"comprehensive_architecture_study_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Setup datasets
    train_transform, eval_transform = get_transforms()
    
    print("\n📂 Creating datasets...")
    full_dataset = MicrobeadDataset(
        image_dir, density_file,
        transform=train_transform,
        data_percentage=args.data_percentage,
        dilution_factors=args.dilution_factors,
        use_all_dilutions=args.use_all_dilutions
    )
    
    test_dataset = MicrobeadDataset(
        image_dir, density_file,
        transform=eval_transform,
        data_percentage=args.data_percentage,
        dilution_factors=args.dilution_factors,
        use_all_dilutions=args.use_all_dilutions
    )
    
    # Create data splits
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    print(f"📊 Dataset splits: {train_size} train, {val_size} val, {test_size} test")
    
    # Create data loaders
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create data loaders with adaptive settings (will be updated per model)
    base_batch_size = args.base_batch_size
    base_num_workers = args.base_num_workers
    
    train_loader = DataLoader(full_dataset, batch_size=base_batch_size, sampler=train_sampler,
                             num_workers=base_num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(full_dataset, batch_size=base_batch_size, sampler=val_sampler,
                           num_workers=base_num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=base_batch_size, sampler=test_sampler,
                            num_workers=base_num_workers, pin_memory=True)
    
    # Define all architectures to test
    all_architectures = []
    
    # Baseline architectures
    if args.run_baselines:
        all_architectures.extend([
            BaselineShallowCNN(base_filters=64),
            BaselineDeepCNN(base_filters=32)
        ])
    
    # ResNet architectures
    if args.run_resnets:
        all_architectures.extend([
            ResNetShallowCNN(base_filters=64),
            ResNetDeepCNN(base_filters=32)
        ])
    
    # UNet architectures (multiple configurations)
    if args.run_unets:
        unet_configs = [
            {'base_filters': 32, 'skip_connection_type': 'channel_reduced', 'max_filters': 128},
            {'base_filters': 36, 'skip_connection_type': 'channel_reduced', 'max_filters': 128},
            {'base_filters': 32, 'skip_connection_type': 'full_concat', 'max_filters': 96},
            {'base_filters': 36, 'skip_connection_type': 'full_concat', 'max_filters': 128},
            {'base_filters': 40, 'skip_connection_type': 'channel_reduced', 'max_filters': 144},
        ]
        for config in unet_configs:
            all_architectures.append(UNetCNN(**config))
    
    # DenseNet architecture
    if args.run_densenet:
        all_architectures.append(DenseNetStyleCNN(base_filters=64))
    
    print(f"\n🧪 Testing {len(all_architectures)} architectural configurations")
    
    # Run comprehensive study
    all_results = []
    study_start_time = time.time()
    
    for exp_num, model in enumerate(all_architectures, 1):
        print(f"\n" + "="*80)
        print(f"🔬 EXPERIMENT {exp_num}/{len(all_architectures)}: {model.name}")
        print(f"   Architecture Type: {getattr(model, 'architecture_type', 'Unknown')}")
        print(f"   Depth: {model.depth} layers")
        print(f"   Skip Connections: {model.has_skip_connections}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("="*80)
        
        experiment_start_time = time.time()
        
        # Training configuration
        training_config = {
            'num_epochs': args.epochs,
            'patience': args.patience,
            'learning_rate': args.learning_rate,
            'mixed_precision': args.mixed_precision,
            'track_gradients': args.track_gradients,
            'output_dir': output_dir
        }
        
        try:
            # Get architecture-specific configuration
            arch_config = get_architecture_config(model)
            
            # Create adaptive data loaders for this specific architecture
            adaptive_train_loader = DataLoader(
                full_dataset, 
                batch_size=arch_config['batch_size'], 
                sampler=train_sampler,
                num_workers=arch_config['num_workers'], 
                pin_memory=True, 
                persistent_workers=True if arch_config['num_workers'] > 0 else False
            )
            
            adaptive_val_loader = DataLoader(
                full_dataset, 
                batch_size=arch_config['batch_size'], 
                sampler=val_sampler,
                num_workers=arch_config['num_workers'], 
                pin_memory=True, 
                persistent_workers=True if arch_config['num_workers'] > 0 else False
            )
            
            # Update training config with architecture-specific settings
            training_config.update(arch_config)
            
            # Train model
            trained_model, training_stats, gradient_tracker = train_model_comprehensive(
                model, adaptive_train_loader, adaptive_val_loader, training_config, device
            )
            
            # Evaluate model
            eval_metrics = evaluate_model_comprehensive(
                trained_model, test_loader, training_config, device
            )
            
            if eval_metrics is None:
                print(f"❌ Experiment {exp_num} evaluation failed")
                continue
            
            experiment_time = (time.time() - experiment_start_time) / 60
            
            # Combine results
            experiment_results = {
                'experiment_number': exp_num,
                'training': training_stats,
                'evaluation': eval_metrics,
                'experiment_time_minutes': experiment_time
            }
            
            all_results.append(experiment_results)
            
            # Save individual results
            result_file = os.path.join(output_dir, f'experiment_{exp_num}_{model.name}_results.json')
            with open(result_file, 'w') as f:
                json.dump(fix_json_serialization(experiment_results), f, indent=4)
            
            print(f"✅ Experiment {exp_num} completed in {experiment_time:.2f} minutes")
            print(f"📈 Performance: R² = {eval_metrics['performance_metrics']['r2_score']:.6f}")
            
            # Cleanup
            del trained_model
            enhanced_memory_cleanup(aggressive=True)
            
        except Exception as e:
            print(f"❌ Experiment {exp_num} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Save error info
            error_info = {
                'experiment_number': exp_num,
                'model_name': model.name,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            error_file = os.path.join(output_dir, f'experiment_{exp_num}_{model.name}_ERROR.json')
            with open(error_file, 'w') as f:
                json.dump(error_info, f, indent=4)
            
            enhanced_memory_cleanup(aggressive=True)
            continue
    
    total_study_time = (time.time() - study_start_time) / 60
    
    print(f"\n🎉 COMPREHENSIVE ARCHITECTURE STUDY COMPLETED!")
    print(f"⏱️  Total execution time: {total_study_time:.2f} minutes")
    print(f"✅ Successful experiments: {len(all_results)}/{len(all_architectures)}")
    
    if len(all_results) >= 3:
        print("\n📊 Performing comprehensive statistical analysis...")
        
        # Statistical analysis
        statistical_results, comparison_df = perform_comprehensive_statistical_analysis(all_results, output_dir)
        
        # Save complete results
        complete_results = {
            'study_info': {
                'study_name': 'Comprehensive CNN Architecture Study',
                'completion_time': datetime.now().isoformat(),
                'total_time_minutes': total_study_time,
                'successful_experiments': len(all_results),
                'total_architectures_tested': len(all_architectures)
            },
            'experimental_results': fix_json_serialization(all_results),
            'statistical_analysis': statistical_results,
            'comparison_summary': fix_json_serialization(comparison_df.to_dict('records'))
        }
        
        with open(os.path.join(output_dir, 'complete_comprehensive_study.json'), 'w') as f:
            json.dump(complete_results, f, indent=4)
        
        # Summary
        print("\n" + "="*80)
        print("📋 STUDY SUMMARY")
        print("="*80)
        
        if not comparison_df.empty:
            best_model = comparison_df.loc[comparison_df['R2_Score'].idxmax()]
            print(f"🏆 Best Architecture: {best_model['Model']}")
            print(f"   Architecture Type: {best_model['Architecture_Type']}")
            print(f"   R² Score: {best_model['R2_Score']:.6f}")
            print(f"   Skip Connections: {best_model['Has_Skip_Connections']}")
            print(f"   Parameters: {best_model['Parameters']:,}")
            print(f"   Training Time: {best_model['Training_Time_Min']:.2f} minutes")
        
        print(f"\n📁 Complete results saved to: {output_dir}")
        
    else:
        print(f"\n⚠️ Insufficient successful experiments ({len(all_results)}/{len(all_architectures)}) for statistical analysis")
    
    # Final cleanup
    enhanced_memory_cleanup(aggressive=True)
    print("\n🔬 Comprehensive CNN Architecture Study Complete!")

if __name__ == "__main__":
    main()
