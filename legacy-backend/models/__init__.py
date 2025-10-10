"""
Deep Learning Models for Brain Tumor Detection

This module contains state-of-the-art neural network architectures for:
- Advanced tumor classification with uncertainty quantification
- High-precision tumor segmentation with attention mechanisms
- Multi-modal analysis with transformer architectures
- Ensemble models for improved robustness

Enhanced with modern features:
- Attention mechanisms (spatial & channel)
- Deep supervision
- Uncertainty quantification
- Medical-specific Vision Transformers
- Advanced loss functions

Author: Brain MRI Tumor Detector Team
Date: October 8, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Import enhanced models
try:
    from .enhanced_models import (
        AdvancedUNet3D, MedicalViT3D, EnsembleModel,
        CombinedLoss as EnhancedCombinedLoss, 
        DiceLoss as EnhancedDiceLoss, 
        FocalLoss as EnhancedFocalLoss, 
        BoundaryLoss as EnhancedBoundaryLoss,
        create_enhanced_model
    )
    ENHANCED_MODELS_AVAILABLE = True
    logger.info("Enhanced models loaded successfully")
except ImportError as e:
    ENHANCED_MODELS_AVAILABLE = False
    logger.warning(f"Enhanced models not available: {e}")
    # Set to None for fallback
    AdvancedUNet3D = None
    MedicalViT3D = None
    EnsembleModel = None
    EnhancedCombinedLoss = None
    EnhancedDiceLoss = None
    EnhancedFocalLoss = None
    EnhancedBoundaryLoss = None
    create_enhanced_model = None

# Legacy TensorFlow support (optional)
tf = None
keras = None  
layers = None

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available for legacy models")

# Enhanced model aliases (avoid conflicts with legacy classes)
UNet3DAdvanced = None
ViT3DMedical = None
ModelEnsemble = None
LossCombined = None
LossDice = None
LossFocal = None
LossBoundary = None

if ENHANCED_MODELS_AVAILABLE:
    try:
        UNet3DAdvanced = AdvancedUNet3D
        ViT3DMedical = MedicalViT3D
        ModelEnsemble = EnsembleModel
        LossCombined = EnhancedCombinedLoss
        LossDice = EnhancedDiceLoss
        LossFocal = EnhancedFocalLoss
        LossBoundary = EnhancedBoundaryLoss
    except NameError:
        logger.warning("Some enhanced models could not be loaded")
        pass


class UNet3D(nn.Module):
    """3D U-Net architecture for brain tumor segmentation."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 4, init_features: int = 32):
        """
        Initialize 3D U-Net model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output classes
            init_features: Number of initial features
        """
        super(UNet3D, self).__init__()
        
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet3D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True),
        )


class ResNet3D(nn.Module):
    """3D ResNet for brain tumor classification."""
    
    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        """
        Initialize 3D ResNet model.
        
        Args:
            num_classes: Number of tumor classes
            in_channels: Number of input channels
        """
        super(ResNet3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(self._residual_block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(self._residual_block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _residual_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def create_tensorflow_unet(input_shape: Tuple[int, ...], num_classes: int = 4):
    """
    Create a 3D U-Net model using TensorFlow/Keras.
    Note: This function requires TensorFlow to be installed.
    
    Args:
        input_shape: Input shape (depth, height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model or None if TensorFlow not available
    """
    if not TF_AVAILABLE or tf is None or keras is None or layers is None:
        logger.warning("TensorFlow not available, cannot create TensorFlow U-Net")
        return None
        
    try:
        inputs = keras.Input(shape=input_shape)
        
        # Encoder
        conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
        
        conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
        
        conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)
        
        conv4 = layers.Conv3D(256, 3, activation='relu', padding='same')(pool3)
        conv4 = layers.Conv3D(256, 3, activation='relu', padding='same')(conv4)
        
        # Decoder
        up5 = layers.UpSampling3D(size=(2, 2, 2))(conv4)
        up5 = layers.Conv3D(128, 2, activation='relu', padding='same')(up5)
        merge5 = layers.concatenate([conv3, up5], axis=4)
        conv5 = layers.Conv3D(128, 3, activation='relu', padding='same')(merge5)
        conv5 = layers.Conv3D(128, 3, activation='relu', padding='same')(conv5)
        
        up6 = layers.UpSampling3D(size=(2, 2, 2))(conv5)
        up6 = layers.Conv3D(64, 2, activation='relu', padding='same')(up6)
        merge6 = layers.concatenate([conv2, up6], axis=4)
        conv6 = layers.Conv3D(64, 3, activation='relu', padding='same')(merge6)
        conv6 = layers.Conv3D(64, 3, activation='relu', padding='same')(conv6)
        
        up7 = layers.UpSampling3D(size=(2, 2, 2))(conv6)
        up7 = layers.Conv3D(32, 2, activation='relu', padding='same')(up7)
        merge7 = layers.concatenate([conv1, up7], axis=4)
        conv7 = layers.Conv3D(32, 3, activation='relu', padding='same')(merge7)
        conv7 = layers.Conv3D(32, 3, activation='relu', padding='same')(conv7)
        
        outputs = layers.Conv3D(num_classes, 1, activation='softmax')(conv7)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error creating TensorFlow U-Net: {e}")
        return None


class AttentionUNet3D(nn.Module):
    """3D U-Net with attention mechanism for improved tumor detection."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 4):
        super(AttentionUNet3D, self).__init__()
        # Implementation would include attention gates
        # This is a simplified version
        self.unet = UNet3D(in_channels, out_channels)
        
    def forward(self, x):
        return self.unet(x)


class MultiModalCNN(nn.Module):
    """Multi-modal CNN for processing different MRI sequences."""
    
    def __init__(self, num_modalities: int = 4, num_classes: int = 4):
        """
        Initialize multi-modal CNN.
        
        Args:
            num_modalities: Number of MRI modalities (T1, T1c, T2, FLAIR)
            num_classes: Number of tumor classes
        """
        super(MultiModalCNN, self).__init__()
        
        # Individual feature extractors for each modality
        self.feature_extractors = nn.ModuleList([
            self._create_feature_extractor() for _ in range(num_modalities)
        ])
        
        # Fusion layer
        self.fusion = nn.Conv3d(512 * num_modalities, 512, kernel_size=1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def _create_feature_extractor(self):
        """Create a feature extractor for one modality."""
        return nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(256, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
        )
        
    def forward(self, x_list):
        """
        Forward pass with list of modality inputs.
        
        Args:
            x_list: List of tensors for each modality
        """
        features = []
        for i, x in enumerate(x_list):
            feat = self.feature_extractors[i](x)
            features.append(feat)
        
        # Concatenate features from all modalities
        fused = torch.cat(features, dim=1)
        fused = self.fusion(fused)
        
        # Classification
        output = self.classifier(fused)
        return output


def get_model(model_name: str, **kwargs):
    """
    Factory function to get model by name.
    
    Args:
        model_name: Name of the model
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
    """
    models = {
        'unet3d': UNet3D,
        'resnet3d': ResNet3D,
        'attention_unet3d': AttentionUNet3D,
        'multimodal_cnn': MultiModalCNN
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name](**kwargs)


# Loss functions for medical image segmentation
class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1, gamma: float = 2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()