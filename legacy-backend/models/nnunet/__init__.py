#!/usr/bin/env python3
"""
nnU-Net Integration Module for Brain MRI Tumor Detection.

This module provides integration with the nnU-Net framework for
state-of-the-art medical image segmentation.

Author: Brain MRI Tumor Detector Team
Date: October 5, 2025
"""

from .nnunet_model import nnUNetWrapper
from .nnunet_trainer import nnUNetTrainer
from .nnunet_predictor import nnUNetPredictor
from .config import nnUNetConfig

__all__ = [
    'nnUNetWrapper',
    'nnUNetTrainer', 
    'nnUNetPredictor',
    'nnUNetConfig'
]

__version__ = '0.1.0'
__author__ = 'Brain MRI Tumor Detector Team'
__description__ = 'nnU-Net integration for medical image segmentation'