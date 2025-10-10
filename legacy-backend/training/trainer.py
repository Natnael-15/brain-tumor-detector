#!/usr/bin/env python3
"""
Advanced Trainer Module for Phase 1 Models.

This module provides sophisticated training capabilities for:
- Vision Transformers (MedViT3D)
- Hybrid CNN-Transformer models
- Advanced optimization strategies
- Medical-specific training protocols

Author: Brain MRI Tumor Detector Team
Date: October 5, 2025
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim import lr_scheduler
    import wandb
    from monai.losses import DiceLoss, FocalLoss
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch or related libraries not available")


class AdvancedTrainer:
    """
    Advanced trainer for Phase 1 deep learning models.
    
    Supports:
    - Vision Transformers and hybrid models
    - Mixed precision training
    - Advanced loss functions for medical segmentation
    - Comprehensive metrics and monitoring
    - Learning rate scheduling
    - Model checkpointing
    """
    
    def __init__(self, model: Any, config: Dict[str, Any], device: str = 'cuda'):
        """
        Initialize the advanced trainer.
        
        Args:
            model: The model to train
            config: Training configuration dictionary
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model = model
        self.config = config
        self.device = device
        self.logger = self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        
        # Initialize components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.metrics = {}
        
        if TORCH_AVAILABLE:
            self._setup_training_components()
        
    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging for training.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger(f"{__name__}.{type(self.model).__name__}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _setup_training_components(self) -> None:
        """
        Set up optimizer, scheduler, loss function, and metrics.
        """
        if not TORCH_AVAILABLE:
            return
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        training_config = self.config.get('medvit', {}).get('training', {})
        optimizer_name = training_config.get('optimizer', 'adamw').lower()
        lr = training_config.get('learning_rate', 1e-4)
        weight_decay = training_config.get('weight_decay', 1e-2)
        
        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
            
        # Setup scheduler
        scheduler_type = training_config.get('scheduler', 'cosine')
        max_epochs = training_config.get('max_epochs', 200)
        warmup_epochs = training_config.get('warmup_epochs', 10)
        
        if scheduler_type == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs
            )
        elif scheduler_type == 'step':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=10,
                factor=0.5
            )
            
        # Setup loss function
        loss_config = self.config.get('medvit', {}).get('loss', {})
        self._setup_loss_function(loss_config)
        
        # Setup metrics
        self._setup_metrics()
        
    def _setup_loss_function(self, loss_config: Dict[str, Any]) -> None:
        """
        Set up the loss function based on configuration.
        
        Args:
            loss_config: Loss function configuration
        """
        loss_type = loss_config.get('type', 'dice')
        
        if loss_type == 'dice':
            self.criterion = DiceLoss(
                sigmoid=True,
                squared_pred=True,
                reduction='mean'
            )
        elif loss_type == 'focal':
            gamma = loss_config.get('focal_gamma', 2.0)
            self.criterion = FocalLoss(
                gamma=gamma,
                reduction='mean'
            )
        elif loss_type == 'combined':
            dice_weight = loss_config.get('dice_weight', 0.5)
            ce_weight = loss_config.get('ce_weight', 0.5)
            
            dice_loss = DiceLoss(sigmoid=True, squared_pred=True)
            ce_loss = nn.CrossEntropyLoss()
            
            class CombinedLoss(nn.Module):
                def forward(self, pred, target):
                    return dice_weight * dice_loss(pred, target) + ce_weight * ce_loss(pred, target)
                    
            self.criterion = CombinedLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            
    def _setup_metrics(self) -> None:
        """
        Set up evaluation metrics.
        """
        self.metrics = {
            'dice': DiceMetric(
                include_background=False,
                reduction='mean',
                get_not_nans=False
            ),
            'hausdorff': HausdorffDistanceMetric(
                include_background=False,
                percentile=95,
                reduction='mean'
            )
        }
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available for training")
            return 0.0
            
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            gradient_clipping = self.config.get('medvit', {}).get('training', {}).get('gradient_clipping', 1.0)
            if gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clipping)
                
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
                
        return total_loss / num_batches if num_batches > 0 else 0.0
        
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average validation loss, metrics dictionary)
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available for validation")
            return 0.0, {}
            
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()
            
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update metrics
                for metric in self.metrics.values():
                    metric(output, target)
                    
        # Compute final metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics_dict = {}
        
        for name, metric in self.metrics.items():
            try:
                value = metric.aggregate().item()
                metrics_dict[name] = value
            except Exception as e:
                self.logger.warning(f"Could not compute metric {name}: {e}")
                metrics_dict[name] = 0.0
                
        return avg_loss, metrics_dict
        
    def save_checkpoint(self, filepath: str, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        if not TORCH_AVAILABLE:
            return
            
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
            'config': self.config
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            
        self.logger.info(f"Checkpoint saved: {filepath}")
        
    def load_checkpoint(self, filepath: str) -> bool:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not TORCH_AVAILABLE:
            return False
            
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_metric = checkpoint.get('best_metric', 0.0)
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.metrics_history = checkpoint.get('metrics_history', [])
            
            self.logger.info(f"Checkpoint loaded: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False
            
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> bool:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            True if training completed successfully, False otherwise
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available for training")
            return False
            
        try:
            training_config = self.config.get('medvit', {}).get('training', {})
            max_epochs = training_config.get('max_epochs', 200)
            
            # Initialize wandb if configured
            wandb_config = self.config.get('logging', {}).get('wandb', {})
            if wandb_config.get('enabled', False):
                wandb.init(
                    project=wandb_config.get('project', 'brain-tumor-medvit'),
                    entity=wandb_config.get('entity', 'brain-tumor-detector'),
                    config=self.config
                )
                
            self.logger.info(f"Starting training for {max_epochs} epochs")
            start_time = time.time()
            
            for epoch in range(self.current_epoch, max_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Training phase
                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                
                # Validation phase
                val_loss, metrics = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                self.metrics_history.append(metrics)
                
                # Update learning rate
                if self.scheduler:
                    if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(metrics.get('dice', 0.0))
                    else:
                        self.scheduler.step()
                        
                # Log epoch results
                epoch_time = time.time() - epoch_start_time
                self.logger.info(
                    f"Epoch {epoch+1}/{max_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Dice: {metrics.get('dice', 0.0):.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Log to wandb
                if wandb_config.get('enabled', False):
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        **metrics
                    })
                    
                # Save checkpoints
                checkpoint_dir = Path('checkpoints')
                checkpoint_dir.mkdir(exist_ok=True)
                
                checkpoint_path = checkpoint_dir / f"model_epoch_{epoch+1}.pth"
                
                # Check if this is the best model
                current_metric = metrics.get('dice', 0.0)
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
                    
                self.save_checkpoint(str(checkpoint_path), is_best)
                
                # Early stopping check
                early_stopping = self.config.get('medvit', {}).get('monitoring', {}).get('early_stopping', {})
                if early_stopping.get('patience', 0) > 0:
                    patience = early_stopping['patience']
                    if len(self.metrics_history) > patience:
                        recent_metrics = [m.get('dice', 0.0) for m in self.metrics_history[-patience:]]
                        if all(m <= self.best_metric * 0.99 for m in recent_metrics):
                            self.logger.info(f"Early stopping at epoch {epoch+1}")
                            break
                            
            total_time = time.time() - start_time
            self.logger.info(f"Training completed in {total_time:.2f}s")
            
            if wandb_config.get('enabled', False):
                wandb.finish()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False