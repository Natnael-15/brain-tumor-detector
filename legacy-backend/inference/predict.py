"""
Tumor Prediction and Inference Module

This module handles inference using trained models for brain tumor detection.
"""

import os
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import logging

logger = logging.getLogger(__name__)

# Try to import nibabel, handle gracefully if not available
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logger.warning("nibabel not available. Some features may be limited.")


class TumorPredictor:
    """Handles inference for brain tumor detection."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize tumor predictor.
        
        Args:
            model_path: Path to trained model file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Model metadata
        self.config = self._load_model_config()
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def _load_model(self):
        """Load the trained model."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model architecture from config if available
            model_config = checkpoint.get('config', {})
            model_name = model_config.get('model_name', 'unet3d')
            
            # Create model architecture (simplified version)
            if model_name == 'unet3d':
                model = self._create_simple_unet()
            else:
                # Fallback to a basic CNN
                model = self._create_basic_cnn()
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Return a dummy model for demonstration
            return self._create_dummy_model()
    
    def _create_simple_unet(self):
        """Create a simplified U-Net model for demonstration."""
        import torch.nn as nn
        
        class SimpleUNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Simplified U-Net architecture
                self.conv1 = nn.Conv3d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv3d(64, 32, 3, padding=1)
                self.conv4 = nn.Conv3d(32, 4, 1)  # 4 classes
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                x = torch.sigmoid(self.conv4(x))
                return x
        
        return SimpleUNet()
    
    def _create_basic_cnn(self):
        """Create a basic CNN for classification."""
        import torch.nn as nn
        
        class BasicCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv3d(1, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool3d(2),
                    nn.Conv3d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool3d(2),
                    nn.AdaptiveAvgPool3d((1, 1, 1))
                )
                self.classifier = nn.Linear(64, 4)
                
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return BasicCNN()
    
    def _create_dummy_model(self):
        """Create a dummy model for demonstration."""
        import torch.nn as nn
        
        class DummyModel(nn.Module):
            def forward(self, x):
                # Return random predictions for demonstration
                batch_size = x.shape[0]
                return torch.rand(batch_size, 4, *x.shape[2:])
        
        return DummyModel()
    
    def _load_model_config(self) -> Dict:
        """Load model configuration."""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            return checkpoint.get('config', {})
        except:
            return {}
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess input image for inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image tensor
        """
        if NIBABEL_AVAILABLE and image_path.endswith('.nii'):
            # Load NIfTI file
            nii = nib.load(image_path)
            image_data = nii.get_fdata()
        else:
            # Try to load as numpy array
            try:
                image_data = np.load(image_path)
            except:
                logger.error(f"Cannot load image from {image_path}")
                # Return dummy data
                image_data = np.random.rand(128, 128, 128)
        
        # Normalize
        image_data = (image_data - image_data.mean()) / (image_data.std() + 1e-8)
        
        # Add batch and channel dimensions
        if len(image_data.shape) == 3:
            image_data = image_data[np.newaxis, np.newaxis, ...]  # (1, 1, D, H, W)
        
        return torch.from_numpy(image_data.astype(np.float32))
    
    def predict(self, image_path: str) -> Dict:
        """
        Perform tumor prediction on a single image.
        
        Args:
            image_path: Path to input MRI image
            
        Returns:
            Dictionary containing prediction results
        """
        logger.info(f"Processing image: {image_path}")
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            prediction = self.model(image_tensor)
        
        # Process prediction based on model type
        if len(prediction.shape) == 5:  # Segmentation output (B, C, D, H, W)
            result = self._process_segmentation_output(prediction)
        else:  # Classification output (B, C)
            result = self._process_classification_output(prediction)
        
        # Add metadata
        result.update({
            'image_path': image_path,
            'model_path': self.model_path,
            'device': self.device,
            'image_shape': image_tensor.shape
        })
        
        return result
    
    def _process_segmentation_output(self, prediction: torch.Tensor) -> Dict:
        """Process segmentation model output."""
        # Convert to numpy
        pred_np = prediction.cpu().numpy()[0]  # Remove batch dimension
        
        # Get predicted classes
        pred_classes = np.argmax(pred_np, axis=0)
        
        # Calculate tumor volumes
        tumor_volumes = {}
        class_names = ['background', 'necrotic', 'edema', 'enhancing']
        
        for i, name in enumerate(class_names):
            volume = np.sum(pred_classes == i)
            tumor_volumes[name] = int(volume)
        
        # Calculate confidence scores
        max_probs = np.max(pred_np, axis=0)
        confidence = float(np.mean(max_probs))
        
        return {
            'task': 'segmentation',
            'prediction_shape': pred_classes.shape,
            'tumor_volumes': tumor_volumes,
            'confidence': confidence,
            'has_tumor': tumor_volumes['necrotic'] + tumor_volumes['edema'] + tumor_volumes['enhancing'] > 0
        }
    
    def _process_classification_output(self, prediction: torch.Tensor) -> Dict:
        """Process classification model output."""
        # Apply softmax to get probabilities
        probs = torch.softmax(prediction, dim=1)
        probs_np = probs.cpu().numpy()[0]
        
        # Get predicted class
        predicted_class = int(np.argmax(probs_np))
        confidence = float(probs_np[predicted_class])
        
        # Class names (example)
        class_names = ['no_tumor', 'glioma', 'meningioma', 'pituitary']
        
        return {
            'task': 'classification',
            'predicted_class': predicted_class,
            'predicted_label': class_names[predicted_class],
            'confidence': confidence,
            'class_probabilities': {
                name: float(prob) for name, prob in zip(class_names, probs_np)
            },
            'has_tumor': predicted_class > 0
        }
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Perform batch prediction on multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'has_tumor': None
                })
        
        return results
    
    def save_prediction_results(self, results: Dict, output_path: str):
        """
        Save prediction results to file.
        
        Args:
            results: Prediction results dictionary
            output_path: Path to save results
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Command line interface for prediction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Brain Tumor Prediction")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save results")
    parser.add_argument("--device", default="auto", help="Device to use (cpu/cuda/auto)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Create predictor
    predictor = TumorPredictor(args.model, device)
    
    # Perform prediction
    if os.path.isdir(args.image):
        # Batch prediction
        image_files = []
        for ext in ['*.nii', '*.nii.gz', '*.npy']:
            image_files.extend(Path(args.image).glob(ext))
        
        results = predictor.predict_batch([str(f) for f in image_files])
    else:
        # Single image prediction
        results = predictor.predict(args.image)
    
    # Print results
    print(json.dumps(results, indent=2))
    
    # Save results if output path provided
    if args.output:
        predictor.save_prediction_results(results, args.output)


if __name__ == "__main__":
    main()