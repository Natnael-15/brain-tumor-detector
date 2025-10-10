#!/usr/bin/env python3
"""
Brain MRI Tumor Detection Models Module - Phase 1.

This module provides integration for advanced AI models including:
- nnU-Net for medical image segmentation
- Vision Transformers for 3D medical analysis
- Hybrid architectures

Author: Brain MRI Tumor Detector Team
Date: October 5, 2025
"""

import logging
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Try importing TensorFlow (optional)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available")

# Try importing PyTorch (optional)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

# Import Phase 1 advanced models with graceful fallback
try:
    from .nnunet import nnUNetWrapper, nnUNetTrainer
    NNUNET_AVAILABLE = True
except ImportError as e:
    NNUNET_AVAILABLE = False
    logger.warning(f"nnU-Net not available: {e}")

try:
    from .transformers import MedViT3D, HybridMedViT
    VIT_AVAILABLE = True
except ImportError as e:
    VIT_AVAILABLE = False
    logger.warning(f"Vision Transformers not available: {e}")

# Check if we have the preprocessing dependency issue
try:
    import sys
    sys.path.insert(0, '.')
    from src.data.brats import BraTSDataset
    BRATS_AVAILABLE = True
except ImportError as e:
    BRATS_AVAILABLE = False
    logger.warning(f"BraTS dataset not available: {e}")


def get_model_status() -> Dict[str, bool]:
    """
    Get status of all available model components.
    
    Returns:
        Dictionary with availability status
    """
    return {
        'tensorflow': TF_AVAILABLE,
        'pytorch': TORCH_AVAILABLE,
        'nnunet': NNUNET_AVAILABLE,
        'vision_transformers': VIT_AVAILABLE,
        'brats_dataset': BRATS_AVAILABLE
    }


def print_phase1_status():
    """Print Phase 1 implementation status."""
    print(" Phase 1 Advanced AI Models Status")
    print("=" * 50)
    
    status = get_model_status()
    
    for component, available in status.items():
        indicator = "" if available else "❌"
        print(f"{indicator} {component.replace('_', ' ').title()}")
    
    print("\n📋 Phase 1 Components:")
    print("  • nnU-Net Framework Integration")
    print("  • Medical Vision Transformers (MedViT3D)")
    print("  • Hybrid CNN-Transformer Models")
    print("  • BraTS Dataset Pipeline")
    print("  • Advanced Training Infrastructure")
    
    if not any(status.values()):
        print("\n⚠️  Install dependencies with: pip install -r requirements-phase1.txt")
    else:
        print(f"\n🎯 {sum(status.values())}/{len(status)} components available")


# Export available components
__all__ = ['get_model_status', 'print_phase1_status']

# Add available models to exports
if NNUNET_AVAILABLE:
    __all__.extend(['nnUNetWrapper', 'nnUNetTrainer'])
    
if VIT_AVAILABLE:
    __all__.extend(['MedViT3D', 'HybridMedViT'])


# For backward compatibility, keep the basic models import
try:
    from .basic_models import BasicCNN, create_basic_model
    __all__.extend(['BasicCNN', 'create_basic_model'])
except ImportError:
    # Create a simple fallback
    class BasicCNN:
        def __init__(self, *args, **kwargs):
            print("BasicCNN placeholder - install dependencies for full functionality")
    
    def create_basic_model(*args, **kwargs):
        return BasicCNN()
    
    __all__.extend(['BasicCNN', 'create_basic_model'])


if __name__ == "__main__":
    print_phase1_status()