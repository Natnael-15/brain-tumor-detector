#!/usr/bin/env python3
"""
nnU-Net Model Wrapper for Brain MRI Tumor Detection.

This module provides a wrapper around the nnU-Net framework
for seamless integration with our brain tumor detection pipeline.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

try:
    # Try to import nnU-Net (will be installed later)
    from nnunet.inference.predict import predict_from_folder
    from nnunet.paths import default_plans_identifier
    NNUNET_AVAILABLE = True
except ImportError:
    NNUNET_AVAILABLE = False
    logging.warning("nnU-Net not available. Install with: pip install nnunet")


class nnUNetWrapper:
    """
    Wrapper class for nnU-Net integration.
    
    This class provides a unified interface to nnU-Net functionality
    while maintaining compatibility with our existing pipeline.
    """
    
    def __init__(self, 
                 model_folder: str,
                 folds: Union[str, List[int]] = 'all',
                 mixed_precision: bool = True,
                 checkpoint_name: str = 'model_final_checkpoint'):
        """
        Initialize nnU-Net wrapper.
        
        Args:
            model_folder: Path to trained nnU-Net model
            folds: Which folds to use for prediction ('all' or list of integers)
            mixed_precision: Whether to use mixed precision inference
            checkpoint_name: Name of checkpoint file to use
        """
        self.model_folder = Path(model_folder)
        self.folds = folds
        self.mixed_precision = mixed_precision
        self.checkpoint_name = checkpoint_name
        self.is_initialized = False
        
        if not NNUNET_AVAILABLE:
            raise ImportError("nnU-Net is not installed. Please install with: pip install nnunet")
            
        # Validate model folder exists
        if not self.model_folder.exists():
            raise ValueError(f"Model folder {model_folder} does not exist")
            
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> bool:
        """
        Initialize the nnU-Net model for inference.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Verify model files exist
            plans_file = self.model_folder / 'plans.pkl'
            if not plans_file.exists():
                self.logger.error(f"Plans file not found: {plans_file}")
                return False
                
            # Check for fold directories
            if self.folds == 'all':
                fold_dirs = [d for d in self.model_folder.iterdir() 
                           if d.is_dir() and d.name.startswith('fold_')]
                if not fold_dirs:
                    self.logger.error("No fold directories found")
                    return False
                    
            self.is_initialized = True
            self.logger.info("nnU-Net model initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize nnU-Net: {e}")
            return False
            
    def predict(self, 
                input_folder: str,
                output_folder: str,
                save_npz: bool = False,
                num_threads_preprocessing: int = 6,
                num_threads_nifti_save: int = 2) -> Dict[str, np.ndarray]:
        """
        Run nnU-Net prediction on input data.
        
        Args:
            input_folder: Folder containing input NIfTI files
            output_folder: Folder to save predictions
            save_npz: Whether to save predictions as npz files
            num_threads_preprocessing: Number of threads for preprocessing
            num_threads_nifti_save: Number of threads for saving results
            
        Returns:
            Dict containing prediction results
        """
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("nnU-Net model not initialized")
                
        try:
            # Create output directory
            os.makedirs(output_folder, exist_ok=True)
            
            # Run nnU-Net prediction
            predict_from_folder(
                model=str(self.model_folder),
                input_folder=input_folder,
                output_folder=output_folder,
                folds=self.folds,
                save_npz=save_npz,
                num_threads_preprocessing=num_threads_preprocessing,
                num_threads_nifti_save=num_threads_nifti_save,
                lowres_segmentations=None,
                part_id=0,
                num_parts=1,
                tta=True,  # Test time augmentation
                mixed_precision=self.mixed_precision,
                overwrite_existing=False,
                mode='normal',
                overwrite_all_in_gpu=None,
                step_size=0.5,
                checkpoint_name=self.checkpoint_name
            )
            
            # Load and return results
            results = {}
            output_path = Path(output_folder)
            for file_path in output_path.glob('*.nii.gz'):
                # Load prediction result (placeholder - actual implementation would load NIfTI)
                results[file_path.stem] = np.random.rand(128, 128, 128)  # Placeholder
                
            self.logger.info(f"nnU-Net prediction completed. Results saved to {output_folder}")
            return results
            
        except Exception as e:
            self.logger.error(f"nnU-Net prediction failed: {e}")
            raise
            
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the loaded nnU-Net model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_folder': str(self.model_folder),
            'folds': str(self.folds),
            'mixed_precision': str(self.mixed_precision),
            'checkpoint_name': self.checkpoint_name,
            'initialized': str(self.is_initialized)
        }
        
        if self.model_folder.exists():
            plans_file = self.model_folder / 'plans.pkl'
            info['plans_file_exists'] = str(plans_file.exists())
            
        return info
        
    def preprocess_for_nnunet(self, 
                               image: np.ndarray,
                               output_file: str,
                               spacing: Optional[Tuple[float, float, float]] = None) -> str:
        """
        Preprocess image data for nnU-Net format.
        
        Args:
            image: Input 3D image array
            output_file: Path to save preprocessed file
            spacing: Voxel spacing (x, y, z) in mm
            
        Returns:
            Path to preprocessed file
        """
        try:
            # Ensure image is in correct format
            if image.dtype != np.float32:
                image = image.astype(np.float32)
                
            # nnU-Net expects specific orientation and spacing
            # This is a simplified preprocessing - actual implementation
            # would include proper medical image preprocessing
            
            # Save as NIfTI format (placeholder - actual implementation would use nibabel)
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # For now, save as numpy array (in production, save as NIfTI)
            np.save(output_path.with_suffix('.npy'), image)
            
            self.logger.info(f"Preprocessed image saved to {output_file}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise
            
    @staticmethod
    def download_pretrained_model(task_name: str, 
                                 download_folder: str = './nnunet_models') -> str:
        """
        Download pretrained nnU-Net model.
        
        Args:
            task_name: Name of the nnU-Net task (e.g., 'Task01_BrainTumour')
            download_folder: Folder to download models to
            
        Returns:
            Path to downloaded model
        """
        # This would implement actual model downloading
        # For now, return placeholder path
        model_path = os.path.join(download_folder, task_name)
        os.makedirs(model_path, exist_ok=True)
        
        logging.info(f"Pretrained model would be downloaded to: {model_path}")
        return model_path
        
    def __repr__(self) -> str:
        return f"nnUNetWrapper(model_folder='{self.model_folder}', initialized={self.is_initialized})"


class nnUNetEnsemble:
    """
    Ensemble multiple nnU-Net models for improved performance.
    """
    
    def __init__(self, model_wrappers: List[nnUNetWrapper]):
        """
        Initialize ensemble of nnU-Net models.
        
        Args:
            model_wrappers: List of nnUNetWrapper instances
        """
        self.models = model_wrappers
        self.logger = logging.getLogger(__name__)
        
    def predict_ensemble(self, 
                        input_folder: str,
                        output_folder: str,
                        weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """
        Run ensemble prediction using multiple nnU-Net models.
        
        Args:
            input_folder: Folder containing input data
            output_folder: Folder to save ensemble predictions
            weights: Weights for each model in ensemble
            
        Returns:
            Dictionary containing ensemble prediction results
        """
        if weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)
            
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
            
        # Get predictions from all models
        all_predictions = []
        for i, model in enumerate(self.models):
            model_output = f"{output_folder}_model_{i}"
            predictions = model.predict(input_folder, model_output)
            all_predictions.append(predictions)
            
        # Combine predictions using weighted average
        ensemble_results = {}
        if all_predictions:
            for key in all_predictions[0].keys():
                weighted_sum = np.zeros_like(all_predictions[0][key])
                for pred, weight in zip(all_predictions, weights):
                    weighted_sum += pred[key] * weight
                ensemble_results[key] = weighted_sum
                
        self.logger.info(f"Ensemble prediction completed with {len(self.models)} models")
        return ensemble_results