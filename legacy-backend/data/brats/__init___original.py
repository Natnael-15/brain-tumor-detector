#!/usr/bin/env python3
"""
BraTS Dataset Integration for Brain MRI Tumor Detection.

This module provides integration with the Brain Tumor Segmentation (BraTS)
dataset for training and evaluation of brain tumor detection models.

BraTS is the premier challenge and dataset for brain tumor segmentation,
providing multimodal MRI scans with expert annotations.

Author: Brain MRI Tumor Detector Team
Date: October 5, 2025
"""

from .brats_dataset import BraTSDataset, BraTSDataLoader
from .brats_preprocessing import BraTSPreprocessor
from .brats_evaluation import BraTSEvaluator
from .brats_downloader import BraTSDownloader
from .brats_utils import BraTSUtils

__all__ = [
    'BraTSDataset',
    'BraTSDataLoader',
    'BraTSPreprocessor',
    'BraTSEvaluator',
    'BraTSDownloader',
    'BraTSUtils'
]

__version__ = '0.1.0'
__author__ = 'Brain MRI Tumor Detector Team'
__description__ = 'BraTS dataset integration for brain tumor segmentation'

# BraTS dataset information
BRATS_INFO = {
    'name': 'Brain Tumor Segmentation Challenge',
    'website': 'http://braintumorsegmentation.org/',
    'description': 'Multimodal brain tumor segmentation challenge dataset',
    'modalities': ['T1', 'T1ce', 'T2', 'FLAIR'],
    'labels': {
        0: 'Background',
        1: 'Necrotic and Non-Enhancing Tumor (NCR/NET)',
        2: 'Peritumoral Edema (ED)',
        4: 'GD-enhancing Tumor (ET)'
    },
    'regions': {
        'whole_tumor': [1, 2, 4],
        'tumor_core': [1, 4],
        'enhancing_tumor': [4]
    }
}