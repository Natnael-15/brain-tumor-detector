#!/usr/bin/env python3
"""
nnU-Net Configuration Management.

This module provides configuration utilities for nnU-Net integration.

Author: Brain MRI Tumor Detector Team
Date: October 5, 2025
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class nnUNetConfig:
    """
    Configuration class for nnU-Net settings.
    """
    
    # Model configuration
    task_name: str = "Task501_BrainTumor"
    network: str = "3d_fullres"
    trainer: str = "nnUNetTrainerV2"
    fold: str = "all"
    checkpoint: str = "model_final_checkpoint"
    
    # Data configuration
    input_modalities: Optional[list] = None
    target_spacing: Optional[list] = None
    crop_to_nonzero: bool = True
    normalization: str = "z_score"
    
    # Training configuration
    max_epochs: int = 1000
    batch_size: int = 2
    patch_size: Optional[list] = None
    learning_rate: float = 0.01
    momentum: float = 0.99
    weight_decay: float = 3e-5
    
    # Hardware configuration
    gpu_id: int = 0
    use_cuda: bool = True
    mixed_precision: bool = True
    
    # Paths
    nnunet_raw_data: str = "./data/nnUNet_raw_data"
    nnunet_preprocessed: str = "./data/nnUNet_preprocessed"
    results_folder: str = "./models/nnUNet_results"
    
    def __post_init__(self):
        """Initialize default values for list fields."""
        if self.input_modalities is None:
            self.input_modalities = ["T1", "T1ce", "T2", "FLAIR"]
            
        if self.target_spacing is None:
            self.target_spacing = [1.0, 1.0, 1.0]
            
        if self.patch_size is None:
            self.patch_size = [128, 128, 128]
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'model': {
                'task_name': self.task_name,
                'network': self.network,
                'trainer': self.trainer,
                'fold': self.fold,
                'checkpoint': self.checkpoint
            },
            'data': {
                'input_modalities': self.input_modalities,
                'target_spacing': self.target_spacing,
                'crop_to_nonzero': self.crop_to_nonzero,
                'normalization': self.normalization
            },
            'training': {
                'max_epochs': self.max_epochs,
                'batch_size': self.batch_size,
                'patch_size': self.patch_size,
                'learning_rate': self.learning_rate,
                'momentum': self.momentum,
                'weight_decay': self.weight_decay
            },
            'hardware': {
                'gpu_id': self.gpu_id,
                'use_cuda': self.use_cuda,
                'mixed_precision': self.mixed_precision
            },
            'paths': {
                'nnunet_raw_data': self.nnunet_raw_data,
                'nnunet_preprocessed': self.nnunet_preprocessed,
                'results_folder': self.results_folder
            }
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'nnUNetConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            nnUNetConfig instance
        """
        # Flatten nested dictionary
        flattened = {}
        
        for section, params in config_dict.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    flattened[key] = value
            else:
                flattened[section] = params
                
        return cls(**flattened)
        
    def get_nnunet_env_vars(self) -> Dict[str, str]:
        """
        Get environment variables for nnU-Net.
        
        Returns:
            Dictionary of environment variables
        """
        return {
            'nnUNet_raw_data_base': self.nnunet_raw_data,
            'nnUNet_preprocessed': self.nnunet_preprocessed,
            'RESULTS_FOLDER': self.results_folder
        }