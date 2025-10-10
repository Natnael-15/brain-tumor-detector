"""
MRI Data Preprocessing Module

This module handles the preprocessing of MRI brain scans including:
- DICOM/NIfTI file loading
- Image normalization and standardization
- Skull stripping
- Registration and resampling
- Data augmentation
"""

import os
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path
import logging
from typing import Tuple, Optional, List
import yaml

logger = logging.getLogger(__name__)


class MRIPreprocessor:
    """Handles preprocessing of MRI brain scan data."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the MRI preprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.target_shape = self.config.get('target_shape', (128, 128, 128))
        self.target_spacing = self.config.get('target_spacing', (1.0, 1.0, 1.0))
        
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        return {}
    
    def load_nifti(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load NIfTI file and return image data and affine matrix.
        
        Args:
            file_path: Path to NIfTI file
            
        Returns:
            Tuple of (image_data, affine_matrix)
        """
        try:
            nii = nib.load(file_path)
            return nii.get_fdata(), nii.affine
        except Exception as e:
            logger.error(f"Error loading NIfTI file {file_path}: {str(e)}")
            raise
    
    def normalize_intensity(self, image: np.ndarray, method: str = 'z_score') -> np.ndarray:
        """
        Normalize image intensity values.
        
        Args:
            image: Input image array
            method: Normalization method ('z_score', 'min_max', 'percentile')
            
        Returns:
            Normalized image array
        """
        if method == 'z_score':
            mean = np.mean(image)
            std = np.std(image)
            return (image - mean) / (std + 1e-8)
        
        elif method == 'min_max':
            min_val = np.min(image)
            max_val = np.max(image)
            return (image - min_val) / (max_val - min_val + 1e-8)
        
        elif method == 'percentile':
            p1, p99 = np.percentile(image, [1, 99])
            image = np.clip(image, p1, p99)
            return (image - p1) / (p99 - p1 + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def skull_strip(self, image: np.ndarray) -> np.ndarray:
        """
        Simple skull stripping using thresholding and morphological operations.
        
        Args:
            image: Input brain image
            
        Returns:
            Skull-stripped image
        """
        # Normalize to 0-255 range
        normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Apply Otsu thresholding
        _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        mask = binary.astype(bool)
        result = image.copy()
        result[~mask] = 0
        
        return result
    
    def resample_image(self, image: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Resample image to target shape using interpolation.
        
        Args:
            image: Input image
            target_shape: Target dimensions
            
        Returns:
            Resampled image
        """
        if len(image.shape) == 3:
            # For 3D images, resize each slice
            resized_slices = []
            for i in range(image.shape[2]):
                slice_2d = cv2.resize(image[:, :, i], 
                                    (target_shape[1], target_shape[0]), 
                                    interpolation=cv2.INTER_LINEAR)
                resized_slices.append(slice_2d)
            
            # Stack slices and resize in z-direction if needed
            volume = np.stack(resized_slices, axis=2)
            if volume.shape[2] != target_shape[2]:
                # Simple linear interpolation for z-direction
                z_indices = np.linspace(0, volume.shape[2] - 1, target_shape[2])
                volume_resized = np.zeros(target_shape)
                for i, z_idx in enumerate(z_indices):
                    if z_idx == int(z_idx):
                        volume_resized[:, :, i] = volume[:, :, int(z_idx)]
                    else:
                        # Linear interpolation between adjacent slices
                        z_low = int(np.floor(z_idx))
                        z_high = int(np.ceil(z_idx))
                        weight = z_idx - z_low
                        volume_resized[:, :, i] = (1 - weight) * volume[:, :, z_low] + weight * volume[:, :, z_high]
                return volume_resized
            return volume
        else:
            return cv2.resize(image, target_shape[:2], interpolation=cv2.INTER_LINEAR)
    
    def augment_data(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply data augmentation techniques.
        
        Args:
            image: Input image
            
        Returns:
            List of augmented images
        """
        augmented = [image]  # Original image
        
        # Rotation
        for angle in [90, 180, 270]:
            if len(image.shape) == 3:
                rotated = np.rot90(image, k=angle//90, axes=(0, 1))
            else:
                rotated = np.rot90(image, k=angle//90)
            augmented.append(rotated)
        
        # Flipping
        augmented.append(np.fliplr(image))
        if len(image.shape) == 3:
            augmented.append(np.flipud(image))
        
        # Gaussian noise
        noise = np.random.normal(0, 0.01, image.shape)
        augmented.append(image + noise)
        
        return augmented
    
    def preprocess_single_image(self, image_path: str, output_path: str = None) -> np.ndarray:
        """
        Preprocess a single MRI image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save processed image
            
        Returns:
            Preprocessed image array
        """
        logger.info(f"Preprocessing image: {image_path}")
        
        # Load image
        image_data, affine = self.load_nifti(image_path)
        
        # Normalize intensity
        normalized = self.normalize_intensity(image_data, method='percentile')
        
        # Skull stripping (optional, can be skipped for some datasets)
        if self.config.get('skull_strip', False):
            normalized = self.skull_strip(normalized)
        
        # Resample to target shape
        processed = self.resample_image(normalized, self.target_shape)
        
        # Save processed image if output path provided
        if output_path:
            output_nii = nib.Nifti1Image(processed, affine)
            nib.save(output_nii, output_path)
            logger.info(f"Saved processed image to: {output_path}")
        
        return processed
    
    def process_directory(self, input_dir: str, output_dir: str):
        """
        Process all MRI images in a directory.
        
        Args:
            input_dir: Input directory containing MRI files
            output_dir: Output directory for processed files
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all NIfTI files
        nifti_files = list(input_path.glob("*.nii*"))
        
        logger.info(f"Found {len(nifti_files)} NIfTI files to process")
        
        for file_path in nifti_files:
            try:
                output_file = output_path / f"processed_{file_path.name}"
                self.preprocess_single_image(str(file_path), str(output_file))
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        logger.info("Preprocessing completed!")


def main():
    """Command line interface for preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MRI Data Preprocessing")
    parser.add_argument("--input", required=True, help="Input directory or file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default="config/preprocessing.yaml", help="Config file")
    
    args = parser.parse_args()
    
    preprocessor = MRIPreprocessor(config_path=args.config)
    
    if os.path.isdir(args.input):
        preprocessor.process_directory(args.input, args.output)
    else:
        output_file = os.path.join(args.output, f"processed_{os.path.basename(args.input)}")
        preprocessor.preprocess_single_image(args.input, output_file)


if __name__ == "__main__":
    main()