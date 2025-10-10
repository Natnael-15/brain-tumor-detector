#!/usr/bin/env python3
"""
Configuration Utilities for Brain MRI Tumor Detector.

This module provides utilities for loading, validating, and managing
configuration files for various components of the system.

Author: Brain MRI Tumor Detector Team
Date: October 5, 2025
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid or unsupported format
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
                
        if config is None:
            config = {}
            
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")
        

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration
        
    Raises:
        ValueError: If file format is unsupported
    """
    config_path = Path(config_path)
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
                
    except Exception as e:
        raise ValueError(f"Error saving config file: {e}")
        

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries recursively.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged
    

def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate configuration against a schema.
    
    Args:
        config: Configuration dictionary to validate
        schema: Schema dictionary defining required structure
        
    Returns:
        True if valid, False otherwise
    """
    try:
        return _validate_recursive(config, schema)
    except Exception:
        return False
        

def _validate_recursive(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Recursively validate configuration against schema.
    
    Args:
        config: Configuration dictionary
        schema: Schema dictionary
        
    Returns:
        True if valid, False otherwise
    """
    for key, expected_type in schema.items():
        if key not in config:
            return False
            
        if isinstance(expected_type, dict):
            if not isinstance(config[key], dict):
                return False
            if not _validate_recursive(config[key], expected_type):
                return False
        elif isinstance(expected_type, type):
            if not isinstance(config[key], expected_type):
                return False
                
    return True
    

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the brain tumor detector.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'name': 'basic_cnn',
            'input_shape': [4, 128, 128, 128],
            'num_classes': 4,
            'dropout': 0.2
        },
        'training': {
            'batch_size': 2,
            'learning_rate': 0.001,
            'epochs': 100,
            'optimizer': 'adam',
            'loss_function': 'dice_loss'
        },
        'data': {
            'modalities': ['t1', 't1ce', 't2', 'flair'],
            'normalize': True,
            'augmentation': True
        },
        'paths': {
            'data_dir': './data',
            'output_dir': './output',
            'models_dir': './models',
            'logs_dir': './logs'
        },
        'device': 'cuda',
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
    

def create_config_template(output_path: Union[str, Path]) -> None:
    """
    Create a configuration template file.
    
    Args:
        output_path: Path where to save the template
    """
    template = {
        '# Brain MRI Tumor Detector Configuration': None,
        '# This is a template configuration file': None,
        '# Modify the values according to your needs': None,
        
        'model': {
            'name': 'basic_cnn',  # Options: basic_cnn, unet, medvit3d, hybrid
            'input_shape': [4, 128, 128, 128],  # [channels, depth, height, width]
            'num_classes': 4,  # Background, NCR/NET, ED, ET
            'dropout': 0.2
        },
        
        'training': {
            'batch_size': 2,
            'learning_rate': 0.001,
            'epochs': 100,
            'optimizer': 'adam',  # Options: adam, adamw, sgd
            'loss_function': 'dice_loss',  # Options: dice_loss, focal_loss, combined
            'validation_split': 0.2,
            'early_stopping_patience': 15
        },
        
        'data': {
            'modalities': ['t1', 't1ce', 't2', 'flair'],
            'normalize': True,
            'augmentation': True,
            'crop_size': [128, 128, 128],
            'spacing': [1.0, 1.0, 1.0]
        },
        
        'paths': {
            'data_dir': './data',
            'output_dir': './output',
            'models_dir': './models',
            'logs_dir': './logs',
            'checkpoints_dir': './checkpoints'
        },
        
        'hardware': {
            'device': 'cuda',  # Options: cuda, cpu
            'mixed_precision': True,
            'num_workers': 4
        },
        
        'logging': {
            'level': 'INFO',  # Options: DEBUG, INFO, WARNING, ERROR
            'console': True,
            'file': True,
            'wandb': {
                'enabled': False,
                'project': 'brain-tumor-detector',
                'entity': 'your-wandb-entity'
            }
        },
        
        'evaluation': {
            'metrics': ['dice', 'hausdorff_95', 'sensitivity', 'specificity'],
            'save_predictions': True,
            'visualization': True
        }
    }
    
    # Remove comment keys for actual saving
    clean_template = {k: v for k, v in template.items() if not k.startswith('#')}
    save_config(clean_template, output_path)
    

def load_model_config(model_name: str) -> Dict[str, Any]:
    """
    Load model-specific configuration.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dictionary
    """
    # Define model-specific configurations
    model_configs = {
        'basic_cnn': {
            'architecture': 'cnn',
            'layers': [32, 64, 128, 256],
            'kernel_size': 3,
            'activation': 'relu',
            'batch_norm': True
        },
        'unet': {
            'architecture': 'unet',
            'encoder_channels': [32, 64, 128, 256, 512],
            'decoder_channels': [256, 128, 64, 32],
            'skip_connections': True,
            'attention': False
        },
        'medvit3d': {
            'architecture': 'transformer',
            'patch_size': [16, 16, 16],
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0
        },
        'hybrid': {
            'architecture': 'hybrid',
            'cnn_backbone': 'resnet3d',
            'transformer_layers': 6,
            'fusion_method': 'concatenation'
        }
    }
    
    return model_configs.get(model_name, {})
    

def update_config_paths(config: Dict[str, Any], base_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Update all relative paths in config to be relative to base_path.
    
    Args:
        config: Configuration dictionary
        base_path: Base path for relative paths
        
    Returns:
        Updated configuration dictionary
    """
    base_path = Path(base_path)
    updated_config = config.copy()
    
    if 'paths' in updated_config:
        for key, path in updated_config['paths'].items():
            if not Path(path).is_absolute():
                updated_config['paths'][key] = str(base_path / path)
                
    return updated_config
    

def validate_paths(config: Dict[str, Any], create_missing: bool = True) -> bool:
    """
    Validate that all paths in configuration exist or can be created.
    
    Args:
        config: Configuration dictionary
        create_missing: Whether to create missing directories
        
    Returns:
        True if all paths are valid, False otherwise
    """
    if 'paths' not in config:
        return True
        
    try:
        for key, path in config['paths'].items():
            path_obj = Path(path)
            
            if key.endswith('_dir'):
                # It's a directory
                if not path_obj.exists() and create_missing:
                    path_obj.mkdir(parents=True, exist_ok=True)
                elif not path_obj.exists():
                    return False
                    
        return True
        
    except Exception:
        return False