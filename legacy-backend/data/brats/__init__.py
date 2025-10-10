#!/usr/bin/env python3
"""
BraTS Dataset Integration Module - Simplified.

This module provides essential BraTS dataset functionality
without complex dependencies.
"""

from .brats_dataset import BraTSDataset

# Mock classes for missing components
class BraTSPreprocessor:
    """Mock BraTS preprocessor."""
    def __init__(self, **kwargs):
        self.is_mock = True

class BraTSDataLoader:
    """Simple BraTS data loader."""
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
    def __iter__(self):
        for i in range(len(self.dataset)):
            yield self.dataset[i]
                
    def __len__(self):
        return len(self.dataset) // self.batch_size

__all__ = ['BraTSDataset', 'BraTSDataLoader', 'BraTSPreprocessor']