#!/usr/bin/env python3
"""
Vision Transformer Models for Brain MRI Analysis.

This module implements Vision Transformer architectures specifically
designed for medical imaging and brain tumor detection.

Author: Brain MRI Tumor Detector Team
Date: October 5, 2025
"""

from .medical_vit import MedicalViT, MedViT3D
from .attention_utils import AttentionVisualizer
from .transformer_blocks import TransformerEncoder, PatchEmbedding3D
from .hybrid_models import CNNTransformerHybrid

__all__ = [
    'MedicalViT',
    'MedViT3D', 
    'AttentionVisualizer',
    'TransformerEncoder',
    'PatchEmbedding3D',
    'CNNTransformerHybrid'
]

__version__ = '0.1.0'
__author__ = 'Brain MRI Tumor Detector Team'
__description__ = 'Vision Transformer models for medical image analysis'