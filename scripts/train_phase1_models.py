#!/usr/bin/env python3
"""
Advanced Training Script for Phase 1 Models.

This script implements training for:
- nnU-Net models
- Vision Transformer models
- Hybrid CNN-Transformer models

Author: Brain MRI Tumor Detector Team
Date: October 5, 2025
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add legacy-backend path for Phase 1 models
legacy_backend_path = str(project_root / "legacy-backend")
if legacy_backend_path not in sys.path:
    sys.path.insert(0, legacy_backend_path)

# Initialize imports to None first
nnUNetWrapper = None
nnUNetTrainer = None
MedViT3D = None
HybridMedViT = None
BraTSDataset = None
BraTSDataLoader = None
AdvancedTrainer = None
load_config = None
IMPORTS_AVAILABLE = False

# Import project modules
try:
    from src.models.nnunet import nnUNetWrapper, nnUNetTrainer  # type: ignore
    from src.models.transformers import MedViT3D, HybridMedViT  # type: ignore
    from src.data.brats import BraTSDataset, BraTSDataLoader  # type: ignore
    from src.training.trainer import AdvancedTrainer  # type: ignore
    from src.utils.config import load_config  # type: ignore
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some imports not available: {e}")
    logging.warning("Training script will run in demonstration mode")
    IMPORTS_AVAILABLE = False


class Phase1ModelTrainer:
    """
    Advanced trainer for Phase 1 models including nnU-Net and Vision Transformers.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.device = self._setup_device()
        
        # Initialize components
        self.model = None
        self.trainer = None
        self.data_loaders = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")
            
    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging configuration.
        
        Returns:
            Configured logger
        """
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('training.log')
            ]
        )
        
        return logging.getLogger(__name__)
        
    def _setup_device(self) -> str:
        """
        Set up computation device.
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                self.logger.info("Using CPU")
            return device
        except ImportError:
            return 'cpu'
            
    def setup_nnunet_training(self) -> bool:
        """
        Set up nnU-Net training.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            if not IMPORTS_AVAILABLE or nnUNetTrainer is None:
                self.logger.error("Required imports not available for nnU-Net")
                return False
                
            nnunet_config = self.config.get('nnunet', {})
            
            # Initialize nnU-Net trainer
            self.trainer = nnUNetTrainer(
                task_name=nnunet_config.get('model', {}).get('task_name', 'Task501_BrainTumor'),
                fold=nnunet_config.get('model', {}).get('fold', 'all'),
                network=nnunet_config.get('model', {}).get('network', '3d_fullres')
            )
            
            self.logger.info("nnU-Net training setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup nnU-Net training: {e}")
            return False
            
    def setup_vit_training(self) -> bool:
        """
        Set up Vision Transformer training.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            if not IMPORTS_AVAILABLE or (HybridMedViT is None and MedViT3D is None):
                self.logger.error("Required imports not available for ViT")
                return False
                
            vit_config = self.config.get('medvit', {})
            model_config = vit_config.get('model', {})
            
            # Initialize ViT model
            if model_config.get('name') == 'HybridMedViT' and HybridMedViT is not None:
                self.model = HybridMedViT(
                    image_size=tuple(model_config.get('image_size', [128, 128, 128])),
                    in_channels=model_config.get('in_channels', 4),
                    num_classes=model_config.get('num_classes', 4)
                )
            elif MedViT3D is not None:
                self.model = MedViT3D(
                    image_size=tuple(model_config.get('image_size', [128, 128, 128])),
                    patch_size=tuple(model_config.get('patch_size', [16, 16, 16])),
                    in_channels=model_config.get('in_channels', 4),
                    num_classes=model_config.get('num_classes', 4),
                    embed_dim=model_config.get('embed_dim', 768),
                    depth=model_config.get('depth', 12),
                    num_heads=model_config.get('num_heads', 12),
                    task_type=model_config.get('task_type', 'segmentation')
                )
            else:
                self.logger.error("Neither HybridMedViT nor MedViT3D is available")
                return False
                
            self.logger.info(f"Vision Transformer model initialized: {type(self.model).__name__}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup ViT training: {e}")
            return False
            
    def setup_data_loaders(self, data_dir: str) -> bool:
        """
        Set up data loaders for training.
        
        Args:
            data_dir: Path to dataset directory
            
        Returns:
            True if setup successful, False otherwise
        """
        try:
            if not IMPORTS_AVAILABLE or BraTSDataset is None or BraTSDataLoader is None:
                self.logger.error("Required imports not available for data loading")
                return False
                
            # Create datasets
            train_dataset = BraTSDataset(
                data_dir=data_dir,
                split='train',
                modalities=['t1', 't1ce', 't2', 'flair'],
                load_seg=True
            )
            
            val_dataset = BraTSDataset(
                data_dir=data_dir,
                split='val',
                modalities=['t1', 't1ce', 't2', 'flair'],
                load_seg=True
            )
            
            # Create data loaders
            batch_size = self.config.get('training', {}).get('batch_size', 2)
            
            self.data_loaders['train'] = BraTSDataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            )
            
            self.data_loaders['val'] = BraTSDataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
            
            self.logger.info(f"Data loaders initialized: {len(train_dataset)} train, {len(val_dataset)} val")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup data loaders: {e}")
            return False
            
    def train_nnunet(self, data_dir: str) -> bool:
        """
        Train nnU-Net model.
        
        Args:
            data_dir: Path to training data
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            self.logger.info("Starting nnU-Net training...")
            
            if not self.setup_nnunet_training():
                return False
                
            # Run nnU-Net training
            if self.trainer is None:
                self.logger.error("nnU-Net trainer not properly initialized")
                return False
                
            nnunet_config = self.config.get('nnunet', {})
            training_config = nnunet_config.get('training', {})
            
            success = self.trainer.train(
                data_folder=data_dir,
                max_epochs=training_config.get('max_epochs', 1000),
                batch_size=training_config.get('batch_size', 2),
                learning_rate=training_config.get('learning_rate', 0.01)
            )
            
            if success:
                self.logger.info("nnU-Net training completed successfully")
            else:
                self.logger.error("nnU-Net training failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"nnU-Net training error: {e}")
            return False
            
    def train_vit(self, data_dir: str) -> bool:
        """
        Train Vision Transformer model.
        
        Args:
            data_dir: Path to training data
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            self.logger.info("Starting Vision Transformer training...")
            
            if not self.setup_vit_training():
                return False
                
            if not self.setup_data_loaders(data_dir):
                return False
                
            # Initialize advanced trainer
            if AdvancedTrainer is None:
                self.logger.error("AdvancedTrainer not available")
                return False
                
            trainer = AdvancedTrainer(
                model=self.model,
                config=self.config,
                device=self.device
            )
            
            # Train model
            success = trainer.train(
                train_loader=self.data_loaders['train'],
                val_loader=self.data_loaders['val']
            )
            
            if success:
                self.logger.info("Vision Transformer training completed successfully")
            else:
                self.logger.error("Vision Transformer training failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Vision Transformer training error: {e}")
            return False
            
    def run_training(self, model_type: str, data_dir: str) -> bool:
        """
        Run training for specified model type.
        
        Args:
            model_type: Type of model to train ('nnunet', 'vit', 'hybrid')
            data_dir: Path to training data
            
        Returns:
            True if training successful, False otherwise
        """
        self.logger.info(f"Starting {model_type} training...")
        
        if model_type.lower() == 'nnunet':
            return self.train_nnunet(data_dir)
        elif model_type.lower() in ['vit', 'medvit', 'transformer']:
            return self.train_vit(data_dir)
        elif model_type.lower() == 'hybrid':
            # Set config to use hybrid model
            self.config['medvit']['model']['name'] = 'HybridMedViT'
            return self.train_vit(data_dir)
        else:
            self.logger.error(f"Unknown model type: {model_type}")
            return False
            

def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description='Train Phase 1 models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                       choices=['nnunet', 'vit', 'hybrid'],
                       help='Model type to train')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set GPU
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        
    # Initialize trainer
    try:
        trainer = Phase1ModelTrainer(args.config)
        
        # Run training
        success = trainer.run_training(args.model, args.data)
        
        if success:
            print(f"✅ {args.model} training completed successfully!")
            sys.exit(0)
        else:
            print(f"❌ {args.model} training failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Training script error: {e}")
        sys.exit(1)
        

if __name__ == '__main__':
    main()