#!/usr/bin/env python3
"""
Enhanced Deep Learning Models for Brain Tumor Detection

This module provides state-of-the-art neural network architectures optimized
for brain tumor detection and analysis, including:
- Enhanced 3D U-Net with attention mechanisms
- Advanced Vision Transformers for medical imaging
- Multi-scale feature fusion networks
- Uncertainty quantification models

Author: Brain MRI Tumor Detector Team
Date: October 8, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class SpatialAttention3D(nn.Module):
    """3D Spatial Attention Module for medical images."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv3d(in_channels // 8, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling along spatial dimensions
        avg_pool = F.adaptive_avg_pool3d(x, 1)
        max_pool = F.adaptive_max_pool3d(x, 1)
        
        # Apply convolutions
        avg_out = self.conv2(F.relu(self.conv1(avg_pool)))
        max_out = self.conv2(F.relu(self.conv1(max_pool)))
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class ChannelAttention3D(nn.Module):
    """3D Channel Attention Module."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class EnhancedResidualBlock3D(nn.Module):
    """Enhanced 3D Residual Block with attention and improved skip connections."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 use_attention: bool = True, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Attention mechanisms
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = ChannelAttention3D(out_channels)
            self.spatial_attention = SpatialAttention3D(out_channels)
        
        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        self.dropout = nn.Dropout3d(dropout)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip_connection(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        if self.use_attention:
            out = self.channel_attention(out)
            out = self.spatial_attention(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class AdvancedUNet3D(nn.Module):
    """
    Advanced 3D U-Net with attention mechanisms, deep supervision,
    and multi-scale feature fusion for brain tumor segmentation.
    """
    
    def __init__(self, 
                 in_channels: int = 1, 
                 out_channels: int = 4, 
                 base_filters: int = 32,
                 deep_supervision: bool = True,
                 use_attention: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        filters = [base_filters * (2**i) for i in range(5)]  # [32, 64, 128, 256, 512]
        
        # Encoder
        self.encoder1 = self._make_encoder_block(in_channels, filters[0], use_attention, dropout)
        self.encoder2 = self._make_encoder_block(filters[0], filters[1], use_attention, dropout)
        self.encoder3 = self._make_encoder_block(filters[1], filters[2], use_attention, dropout)
        self.encoder4 = self._make_encoder_block(filters[2], filters[3], use_attention, dropout)
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(filters[3], filters[4], use_attention, dropout)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose3d(filters[4], filters[3], 2, 2)
        self.decoder4 = self._make_decoder_block(filters[4], filters[3], use_attention, dropout)
        
        self.upconv3 = nn.ConvTranspose3d(filters[3], filters[2], 2, 2)
        self.decoder3 = self._make_decoder_block(filters[3], filters[2], use_attention, dropout)
        
        self.upconv2 = nn.ConvTranspose3d(filters[2], filters[1], 2, 2)
        self.decoder2 = self._make_decoder_block(filters[2], filters[1], use_attention, dropout)
        
        self.upconv1 = nn.ConvTranspose3d(filters[1], filters[0], 2, 2)
        self.decoder1 = self._make_decoder_block(filters[1], filters[0], use_attention, dropout)
        
        # Output layers
        self.final_conv = nn.Conv3d(filters[0], out_channels, 1)
        
        # Deep supervision outputs
        if deep_supervision:
            self.deep_conv1 = nn.Conv3d(filters[1], out_channels, 1)
            self.deep_conv2 = nn.Conv3d(filters[2], out_channels, 1)
            self.deep_conv3 = nn.Conv3d(filters[3], out_channels, 1)
        
        self.pools = nn.ModuleList([nn.MaxPool3d(2) for _ in range(4)])
        
    def _make_encoder_block(self, in_channels: int, out_channels: int, 
                           use_attention: bool, dropout: float) -> nn.Module:
        return nn.Sequential(
            EnhancedResidualBlock3D(in_channels, out_channels, use_attention=use_attention, dropout=dropout),
            EnhancedResidualBlock3D(out_channels, out_channels, use_attention=use_attention, dropout=dropout)
        )
    
    def _make_decoder_block(self, in_channels: int, out_channels: int,
                           use_attention: bool, dropout: float) -> nn.Module:
        return nn.Sequential(
            EnhancedResidualBlock3D(in_channels, out_channels, use_attention=use_attention, dropout=dropout),
            EnhancedResidualBlock3D(out_channels, out_channels, use_attention=use_attention, dropout=dropout)
        )
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pools[0](enc1))
        enc3 = self.encoder3(self.pools[1](enc2))
        enc4 = self.encoder4(self.pools[2](enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pools[3](enc4))
        
        # Decoder path with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final output
        final_output = self.final_conv(dec1)
        
        if self.deep_supervision and self.training:
            # Deep supervision outputs
            deep1 = F.interpolate(self.deep_conv1(dec2), size=x.shape[2:], mode='trilinear')
            deep2 = F.interpolate(self.deep_conv2(dec3), size=x.shape[2:], mode='trilinear')
            deep3 = F.interpolate(self.deep_conv3(dec4), size=x.shape[2:], mode='trilinear')
            
            return {
                'main': final_output,
                'deep1': deep1,
                'deep2': deep2,
                'deep3': deep3
            }
        
        return final_output


class MedicalViT3D(nn.Module):
    """
    Advanced 3D Vision Transformer specifically designed for medical imaging
    with spatial-aware attention and medical-specific augmentations.
    """
    
    def __init__(self,
                 image_size: Tuple[int, int, int] = (128, 128, 128),
                 patch_size: Tuple[int, int, int] = (16, 16, 16),
                 in_channels: int = 1,
                 num_classes: int = 4,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 task_type: str = 'segmentation'):
        super().__init__()
        
        self.task_type = task_type
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = self._create_patch_embedding(
            image_size, patch_size, in_channels, embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Positional embeddings with 3D awareness
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock3D(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Task-specific heads
        if task_type == 'classification':
            self.head = nn.Linear(embed_dim, num_classes)
        elif task_type == 'segmentation':
            self.seg_head = self._create_segmentation_head(embed_dim, num_classes, image_size, patch_size)
        
        # Initialize weights
        self._init_weights()
        
    def _create_patch_embedding(self, image_size, patch_size, in_channels, embed_dim):
        """Create 3D patch embedding layer."""
        
        class PatchEmbed3D(nn.Module):
            def __init__(self, image_size, patch_size, in_channels, embed_dim):
                super().__init__()
                self.image_size = image_size
                self.patch_size = patch_size
                self.num_patches = (
                    (image_size[0] // patch_size[0]) *
                    (image_size[1] // patch_size[1]) *
                    (image_size[2] // patch_size[2])
                )
                
                self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
                
            def forward(self, x):
                x = self.proj(x)  # (B, embed_dim, D', H', W')
                x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
                return x
                
        return PatchEmbed3D(image_size, patch_size, in_channels, embed_dim)
    
    def _create_segmentation_head(self, embed_dim, num_classes, image_size, patch_size):
        """Create segmentation head for pixel-wise prediction."""
        return nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def _init_weights(self):
        """Initialize model weights."""
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add classification token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        if self.task_type == 'classification':
            # Use CLS token for classification
            cls_output = x[:, 0]
            return self.head(cls_output)
            
        elif self.task_type == 'segmentation':
            # Remove CLS token and process patches for segmentation
            patch_features = x[:, 1:]  # Remove CLS token
            
            # Apply segmentation head
            seg_output = self.seg_head(patch_features)
            
            # Reshape to spatial dimensions
            patch_h = self.patch_embed.image_size[1] // self.patch_embed.patch_size[1]
            patch_w = self.patch_embed.image_size[2] // self.patch_embed.patch_size[2]
            patch_d = self.patch_embed.image_size[0] // self.patch_embed.patch_size[0]
            
            seg_output = seg_output.transpose(1, 2).reshape(
                B, self.num_classes, patch_d, patch_h, patch_w
            )
            
            # Upsample to original resolution
            seg_output = F.interpolate(
                seg_output, size=(D, H, W), mode='trilinear', align_corners=False
            )
            
            return seg_output
        
        else:
            # Default fallback - return classification output
            cls_output = x[:, 0]
            return self.head(cls_output) if hasattr(self, 'head') else x[:, 0]


class TransformerBlock3D(nn.Module):
    """Enhanced Transformer block with 3D spatial awareness."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention3D(embed_dim, num_heads, dropout)
        self.drop_path = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MultiHeadAttention3D(nn.Module):
    """3D-aware multi-head attention for medical imaging."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class EnsembleModel(nn.Module):
    """
    Advanced ensemble model that combines multiple architectures
    with uncertainty quantification.
    """
    
    def __init__(self, models: List[nn.Module], num_classes: int = 4):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
        self.ensemble_weights = nn.Parameter(torch.ones(len(models)) / len(models))
        
        # Uncertainty quantification head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(len(models) * num_classes, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)  # Uncertainty scores
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        predictions = []
        
        # Get predictions from all models
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                if isinstance(pred, dict):
                    pred = pred['main']  # For models with deep supervision
                predictions.append(pred)
        
        # Stack predictions
        stacked_preds = torch.stack(predictions, dim=0)  # (num_models, B, C, ...)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_pred = torch.sum(weights.view(-1, 1, 1, 1, 1, 1) * stacked_preds, dim=0)
        
        # Calculate uncertainty
        pred_variance = torch.var(stacked_preds, dim=0)
        uncertainty = torch.mean(pred_variance, dim=1, keepdim=True)  # Average across classes
        
        # Quantify prediction confidence
        confidence = 1.0 - uncertainty
        
        return {
            'prediction': ensemble_pred,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'individual_predictions': stacked_preds  # Return as tensor instead of list
        }


# Enhanced Loss Functions
class CombinedLoss(nn.Module):
    """Combined loss function for medical image segmentation."""
    
    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.3, 
                 boundary_weight: float = 0.2, num_classes: int = 4):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        self.dice_loss = DiceLoss(num_classes)
        self.focal_loss = FocalLoss()
        self.boundary_loss = BoundaryLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        total_loss = (self.dice_weight * dice + 
                     self.focal_weight * focal + 
                     self.boundary_weight * boundary)
        
        return total_loss


class DiceLoss(nn.Module):
    """Multi-class Dice loss for segmentation."""
    
    def __init__(self, num_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        dice_scores = []
        for i in range(self.num_classes):
            pred_i = pred[:, i]
            target_i = target_one_hot[:, i]
            
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        return 1 - torch.mean(torch.stack(dice_scores))


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """Boundary loss for better edge preservation."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Simplified boundary loss implementation
        # In practice, would use distance transforms
        pred_edges = self._get_edges(F.softmax(pred, dim=1))
        target_edges = self._get_edges(F.one_hot(target, pred.shape[1]).permute(0, 4, 1, 2, 3).float())
        
        return F.mse_loss(pred_edges, target_edges)
    
    def _get_edges(self, x: torch.Tensor) -> torch.Tensor:
        # Simple edge detection using gradient
        grad_x = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        grad_y = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        grad_z = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        
        # Pad to maintain original size
        grad_x = F.pad(grad_x, (0, 0, 0, 0, 0, 1))
        grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0))
        grad_z = F.pad(grad_z, (0, 1, 0, 0, 0, 0))
        
        return grad_x + grad_y + grad_z


def create_enhanced_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create enhanced models.
    
    Args:
        model_type: Type of model to create
        **kwargs: Model-specific arguments
        
    Returns:
        Enhanced model instance
    """
    models = {
        'advanced_unet3d': AdvancedUNet3D,
        'medical_vit3d': MedicalViT3D,
        'ensemble': EnsembleModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)


if __name__ == "__main__":
    # Example usage and testing
    
    # Test Advanced U-Net
    model = AdvancedUNet3D(in_channels=1, out_channels=4, base_filters=32)
    x = torch.randn(1, 1, 128, 128, 128)
    output = model(x)
    print(f"Advanced U-Net output shape: {output.shape if isinstance(output, torch.Tensor) else output['main'].shape}")
    
    # Test Medical ViT
    vit_model = MedicalViT3D(
        image_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        in_channels=1,
        num_classes=4,
        task_type='segmentation'
    )
    vit_output = vit_model(x)
    print(f"Medical ViT output shape: {vit_output.shape}")
    
    print("Enhanced models created successfully!")