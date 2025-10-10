"""
Model Training Module

This module handles the training of deep learning models for brain tumor detection.
Includes training loops, validation, and model checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml
import json
from datetime import datetime

# Try to import models, handle if not available
try:
    from ..models import UNet3D, ResNet3D, DiceLoss, FocalLoss
except ImportError:
    # Fallback for when models module is not available
    UNet3D = ResNet3D = DiceLoss = FocalLoss = None

logger = logging.getLogger(__name__)


class BrainTumorDataset(Dataset):
    """Dataset class for brain tumor MRI data."""
    
    def __init__(self, data_dir: str, transform=None, mode: str = 'train'):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing processed MRI data
            transform: Data augmentation transforms
            mode: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.mode = mode
        
        # Load data file paths
        self.data_files = self._load_file_paths()
        
    def _load_file_paths(self):
        """Load file paths for images and labels."""
        data_files = []
        
        # Look for processed image files
        image_files = list(self.data_dir.glob("*.npy"))
        
        for img_file in image_files:
            # Assume corresponding label file exists
            label_file = img_file.parent / f"label_{img_file.name}"
            if label_file.exists():
                data_files.append({
                    'image': str(img_file),
                    'label': str(label_file)
                })
        
        logger.info(f"Found {len(data_files)} data samples for {self.mode}")
        return data_files
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        """Get a data sample."""
        file_info = self.data_files[idx]
        
        # Load image and label
        image = np.load(file_info['image']).astype(np.float32)
        label = np.load(file_info['label']).astype(np.long)
        
        # Add channel dimension if needed
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]  # Add channel dimension
        
        # Apply transforms if available
        if self.transform:
            image = self.transform(image)
        
        return torch.from_numpy(image), torch.from_numpy(label)


class ModelTrainer:
    """Handles training of brain tumor detection models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize model trainer.
        
        Args:
            config_path: Path to training configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Training parameters
        self.batch_size = self.config.get('batch_size', 4)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.num_epochs = self.config.get('num_epochs', 100)
        self.model_name = self.config.get('model_name', 'unet3d')
        
        # Initialize model
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_loss_function()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        
        # Default configuration
        return {
            'batch_size': 4,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'model_name': 'unet3d',
            'input_channels': 1,
            'num_classes': 4,
            'optimizer': 'adam',
            'loss_function': 'dice'
        }
    
    def _create_model(self):
        """Create and initialize the model."""
        if UNet3D is None:
            logger.error("Model classes not available. Please check imports.")
            raise ImportError("Model classes not found")
        
        if self.model_name == 'unet3d':
            model = UNet3D(
                in_channels=self.config.get('input_channels', 1),
                out_channels=self.config.get('num_classes', 4)
            )
        elif self.model_name == 'resnet3d':
            model = ResNet3D(
                num_classes=self.config.get('num_classes', 4),
                in_channels=self.config.get('input_channels', 1)
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        model = model.to(self.device)
        logger.info(f"Created {self.model_name} model with {self._count_parameters(model)} parameters")
        return model
    
    def _create_optimizer(self):
        """Create optimizer."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_loss_function(self):
        """Create loss function."""
        loss_name = self.config.get('loss_function', 'dice').lower()
        
        if loss_name == 'dice' and DiceLoss is not None:
            return DiceLoss()
        elif loss_name == 'focal' and FocalLoss is not None:
            return FocalLoss()
        elif loss_name == 'crossentropy':
            return nn.CrossEntropyLoss()
        else:
            logger.warning(f"Loss function {loss_name} not available, using CrossEntropyLoss")
            return nn.CrossEntropyLoss()
    
    def _count_parameters(self, model):
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        return total_loss / num_batches
    
    def validate_epoch(self, dataloader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, val_loss: float, save_dir: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, save_path)
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f'New best model saved with validation loss: {val_loss:.6f}')
    
    def train(self, data_dir: str, output_dir: str):
        """
        Main training loop.
        
        Args:
            data_dir: Directory containing training data
            output_dir: Directory to save models and logs
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create datasets
        train_dataset = BrainTumorDataset(
            os.path.join(data_dir, 'train'), 
            mode='train'
        )
        val_dataset = BrainTumorDataset(
            os.path.join(data_dir, 'val'), 
            mode='val'
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Training loop
        for epoch in range(self.num_epochs):
            logger.info(f'Epoch {epoch + 1}/{self.num_epochs}')
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            logger.info(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, str(output_path))
            
            # Save training history
            self._save_training_history(str(output_path))
        
        logger.info("Training completed!")
    
    def _save_training_history(self, output_dir: str):
        """Save training history to JSON file."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)


def main():
    """Command line interface for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Brain Tumor Model Training")
    parser.add_argument("--data", required=True, help="Data directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default="config/training.yaml", help="Config file")
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(config_path=args.config)
    trainer.train(args.data, args.output)


if __name__ == "__main__":
    main()