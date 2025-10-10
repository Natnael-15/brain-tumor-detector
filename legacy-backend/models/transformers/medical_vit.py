#!/usr/bin/env python3
"""
Medical Vision Transformer (MedViT) for Brain MRI Analysis.

Implements Vision Transformer architectures specifically designed
for 3D medical imaging and brain tumor detection tasks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math
import logging
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding3D(nn.Module):
    """
    3D Patch Embedding for medical images.
    
    Converts 3D medical images into patch embeddings suitable
    for transformer processing.
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int, int] = (128, 128, 128),
                 patch_size: Tuple[int, int, int] = (16, 16, 16),
                 in_channels: int = 1,
                 embed_dim: int = 768):
        """
        Initialize 3D patch embedding.
        
        Args:
            image_size: Size of input image (D, H, W)
            patch_size: Size of each patch (D, H, W)
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (
            (image_size[0] // patch_size[0]) * 
            (image_size[1] // patch_size[1]) * 
            (image_size[2] // patch_size[2])
        )
        
        # 3D convolution for patch embedding
        self.projection = nn.Conv3d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, C, D, H, W = x.shape
        
        # Verify input dimensions
        assert D == self.image_size[0] and H == self.image_size[1] and W == self.image_size[2], \
            f"Input size {(D, H, W)} doesn't match expected {self.image_size}"
        assert C == self.in_channels, \
            f"Input channels {C} doesn't match expected {self.in_channels}"
            
        # Apply 3D convolution and flatten
        x = self.projection(x)  # (B, embed_dim, D', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        return x


class MultiHeadAttention3D(nn.Module):
    """
    Multi-head attention for 3D medical images with spatial awareness.
    """
    
    def __init__(self, 
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 dropout: float = 0.1,
                 spatial_bias: bool = True):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            spatial_bias: Whether to use spatial position bias
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.spatial_bias = spatial_bias
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Spatial position bias for 3D patches
        if spatial_bias:
            self.relative_position_bias = nn.Parameter(
                torch.zeros(num_heads, 64, 64, 64)  # Max relative positions
            )
            
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Input tensor (B, N, embed_dim)
            mask: Attention mask (optional)
            
        Returns:
            Output tensor (B, N, embed_dim)
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        # Add spatial bias if enabled
        if self.spatial_bias and hasattr(self, 'relative_position_bias'):
            # Simplified spatial bias (in practice, would compute actual 3D positions)
            bias = self.relative_position_bias[:, :N, :N]
            attn += bias.unsqueeze(0)
            
        # Apply mask if provided
        if mask is not None:
            attn.masked_fill_(mask == 0, float('-inf'))
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block for medical image processing.
    """
    
    def __init__(self,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention3D(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transformer block.
        
        Args:
            x: Input tensor (B, N, embed_dim)
            
        Returns:
            Output tensor (B, N, embed_dim)
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class MedViT3D(nn.Module):
    """
    3D Medical Vision Transformer for brain tumor detection.
    
    Implements a Vision Transformer specifically designed for
    3D medical imaging tasks like brain tumor segmentation.
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
        """
        Initialize MedViT3D model.
        
        Args:
            image_size: Size of input image (D, H, W)
            patch_size: Size of each patch (D, H, W)
            in_channels: Number of input channels
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout probability
            task_type: 'segmentation' or 'classification'
        """
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.task_type = task_type
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(
            image_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Class token (for classification tasks)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Task-specific heads
        if task_type == 'classification':
            self.head = nn.Linear(embed_dim, num_classes)
        elif task_type == 'segmentation':
            # Segmentation decoder
            self.seg_decoder = self._build_segmentation_decoder()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
            
        # Initialize weights
        self._init_weights()
        
    def _build_segmentation_decoder(self) -> nn.Module:
        """
        Build segmentation decoder to reconstruct spatial resolution.
        
        Returns:
            Segmentation decoder module
        """
        # Calculate upsampling factor
        patch_volume = np.prod(self.patch_size)
        
        decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, patch_volume * self.num_classes),
            Rearrange(
                'b n (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)',
                p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2],
                h=self.image_size[0] // self.patch_size[0],
                w=self.image_size[1] // self.patch_size[1],
                d=self.image_size[2] // self.patch_size[2],
                c=self.num_classes
            )
        )
        
        return decoder
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize position embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other parameters
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MedViT3D.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Output tensor (shape depends on task_type)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token for classification
        if self.task_type == 'classification':
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
        # Add position embedding
        if self.task_type == 'classification':
            x = x + self.pos_embed
        else:
            x = x + self.pos_embed[:, 1:, :]  # Skip class token position
            
        # Transformer encoder
        for transformer_block in self.transformer:
            x = transformer_block(x)
            
        x = self.norm(x)
        
        # Task-specific output
        if self.task_type == 'classification':
            # Use class token for classification
            cls_output = x[:, 0]  # (B, embed_dim)
            return self.head(cls_output)  # (B, num_classes)
        elif self.task_type == 'segmentation':
            # Remove class token if present and decode
            patch_tokens = x[:, 1:] if x.shape[1] > self.patch_embed.num_patches else x
            return self.seg_decoder(patch_tokens)  # (B, num_classes, D, H, W)
            
    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract attention maps for visualization.
        
        Args:
            x: Input tensor
            layer_idx: Layer index to extract attention from
            
        Returns:
            Attention maps
        """
        # This would implement attention extraction for visualization
        # Simplified implementation
        with torch.no_grad():
            _ = self.forward(x)
            # In practice, would hook into attention layers
            return torch.zeros(x.shape[0], self.num_heads, 
                             self.patch_embed.num_patches, 
                             self.patch_embed.num_patches)


class MedicalViT(MedViT3D):
    """
    Alias for MedViT3D for backwards compatibility.
    """
    pass


class HybridMedViT(nn.Module):
    """
    Hybrid CNN-Transformer model for medical imaging.
    
    Combines convolutional feature extraction with transformer
    processing for improved performance on medical images.
    """
    
    def __init__(self,
                 image_size: Tuple[int, int, int] = (128, 128, 128),
                 in_channels: int = 1,
                 num_classes: int = 4,
                 cnn_embed_dim: int = 256,
                 transformer_embed_dim: int = 768,
                 transformer_depth: int = 6,
                 num_heads: int = 12):
        """
        Initialize Hybrid MedViT.
        
        Args:
            image_size: Size of input image
            in_channels: Number of input channels
            num_classes: Number of output classes
            cnn_embed_dim: CNN feature dimension
            transformer_embed_dim: Transformer embedding dimension
            transformer_depth: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()
        
        # CNN feature extractor
        self.cnn_backbone = self._build_cnn_backbone(in_channels, cnn_embed_dim)
        
        # Projection to transformer dimension
        self.proj = nn.Linear(cnn_embed_dim, transformer_embed_dim)
        
        # Transformer
        self.transformer = MedViT3D(
            image_size=(image_size[0]//4, image_size[1]//4, image_size[2]//4),
            patch_size=(8, 8, 8),
            in_channels=cnn_embed_dim,
            num_classes=num_classes,
            embed_dim=transformer_embed_dim,
            depth=transformer_depth,
            num_heads=num_heads
        )
        
    def _build_cnn_backbone(self, in_channels: int, embed_dim: int) -> nn.Module:
        """
        Build CNN backbone for feature extraction.
        """
        return nn.Sequential(
            # First conv block
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Second conv block
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Third conv block
            nn.Conv3d(128, embed_dim, 3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Hybrid MedViT.
        
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            Output tensor
        """
        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)
        
        # Transformer processing
        output = self.transformer(cnn_features)
        
        return output