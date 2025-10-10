"""
Data Download Script

This script helps download and prepare brain tumor datasets from various sources.
"""

import os
import requests
import gzip
import tarfile
import zipfile
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Downloads and prepares brain tumor datasets."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize the data downloader."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_sample_data(self):
        """Download sample brain MRI data for demonstration."""
        logger.info("Downloading sample brain MRI data...")
        
        # Sample data sources (these are placeholder URLs)
        sample_urls = [
            {
                'name': 'sample_brain_mri_1.nii.gz',
                'url': 'https://example.com/sample_mri_1.nii.gz',
                'description': 'Normal brain MRI scan'
            },
            {
                'name': 'sample_brain_mri_2.nii.gz', 
                'url': 'https://example.com/sample_mri_2.nii.gz',
                'description': 'Brain MRI with tumor'
            }
        ]
        
        # Create sample data directory
        sample_dir = self.data_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        # Since we can't download from placeholder URLs, create dummy data
        self._create_dummy_data(sample_dir)
        
    def _create_dummy_data(self, output_dir: Path):
        """Create dummy MRI data for demonstration."""
        try:
            import numpy as np
        except ImportError:
            print("NumPy not available. Cannot create dummy data.")
            return
        
        logger.info("Creating dummy MRI data for demonstration...")
        
        # Create dummy brain MRI volumes
        for i in range(3):
            # Generate realistic-looking brain MRI data
            brain_volume = self._generate_brain_volume()
            
            if brain_volume is not None:
                # Save as numpy array
                output_file = output_dir / f"sample_brain_{i+1}.npy"
                np.save(output_file, brain_volume)
                logger.info(f"Created dummy data: {output_file}")
                
                # Create corresponding label if it's a tumor case
                if i > 0:  # Make some samples have tumors
                    tumor_mask = self._generate_tumor_mask(brain_volume.shape)
                    if tumor_mask is not None:
                        label_file = output_dir / f"label_sample_brain_{i+1}.npy"
                        np.save(label_file, tumor_mask)
                        logger.info(f"Created tumor mask: {label_file}")
    
    def _generate_brain_volume(self, shape: tuple = (128, 128, 128)):
        """Generate a realistic-looking brain MRI volume."""
        try:
            import numpy as np
        except ImportError:
            print("NumPy not available. Cannot generate brain volume.")
            return None
        
        # Create base brain structure
        volume = np.zeros(shape)
        
        # Add brain tissue (simplified)
        center = tuple(s // 2 for s in shape)
        
        # Create ellipsoidal brain shape
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        brain_mask = ((x - center[2])**2 / (shape[2]//3)**2 + 
                     (y - center[1])**2 / (shape[1]//3)**2 + 
                     (z - center[0])**2 / (shape[0]//3)**2) <= 1
        
        # Add different tissue types
        volume[brain_mask] = np.random.normal(0.5, 0.1, brain_mask.sum())
        
        # Add some structure (ventricles, etc.)
        ventricle_mask = ((x - center[2])**2 / (shape[2]//8)**2 + 
                         (y - center[1])**2 / (shape[1]//8)**2 + 
                         (z - center[0])**2 / (shape[0]//6)**2) <= 1
        volume[ventricle_mask] = np.random.normal(0.1, 0.05, ventricle_mask.sum())
        
        # Add noise
        volume += np.random.normal(0, 0.02, shape)
        
        # Normalize to 0-1 range
        volume = np.clip(volume, 0, 1)
        
        return volume.astype(np.float32)
    
    def _generate_tumor_mask(self, shape: tuple):
        """Generate a tumor segmentation mask."""
        try:
            import numpy as np
        except ImportError:
            print("NumPy not available. Cannot generate tumor mask.")
            return None
        
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Add tumor region
        center = (shape[0]//2 + 10, shape[1]//2 - 15, shape[2]//2 + 5)
        
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        
        # Create tumor core (class 3)
        core_mask = ((x - center[2])**2 / 8**2 + 
                    (y - center[1])**2 / 8**2 + 
                    (z - center[0])**2 / 6**2) <= 1
        mask[core_mask] = 3
        
        # Create enhancing region (class 4)
        enhancing_mask = ((x - center[2])**2 / 12**2 + 
                         (y - center[1])**2 / 12**2 + 
                         (z - center[0])**2 / 9**2) <= 1
        mask[enhancing_mask & (mask == 0)] = 4
        
        # Create edema (class 2)
        edema_mask = ((x - center[2])**2 / 20**2 + 
                     (y - center[1])**2 / 20**2 + 
                     (z - center[0])**2 / 15**2) <= 1
        mask[edema_mask & (mask == 0)] = 2
        
        return mask
    
    def setup_brats_data_structure(self):
        """Set up directory structure for BraTS dataset."""
        logger.info("Setting up BraTS data structure...")
        
        brats_dir = self.data_dir / "BraTS"
        brats_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        subdirs = ["HGG", "LGG", "validation", "testing"]
        for subdir in subdirs:
            (brats_dir / subdir).mkdir(exist_ok=True)
        
        # Create patient directories (examples)
        for grade in ["HGG", "LGG"]:
            grade_dir = brats_dir / grade
            for i in range(5):  # Create 5 example patient directories
                patient_dir = grade_dir / f"BraTS20_Training_{grade}_{i+1:03d}"
                patient_dir.mkdir(exist_ok=True)
                
                # Create placeholder files
                modalities = ["t1", "t1ce", "t2", "flair", "seg"]
                for modality in modalities:
                    placeholder_file = patient_dir / f"BraTS20_Training_{grade}_{i+1:03d}_{modality}.nii.gz"
                    placeholder_file.touch()
        
        logger.info(f"BraTS structure created in {brats_dir}")
    
    def download_kaggle_dataset(self, dataset_name: str):
        """Download dataset from Kaggle (requires Kaggle API)."""
        try:
            import kaggle  # type: ignore
            logger.info(f"Downloading Kaggle dataset: {dataset_name}")
            
            output_dir = self.data_dir / "kaggle" / dataset_name.replace('/', '_')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(output_dir),
                unzip=True
            )
            
            logger.info(f"Downloaded {dataset_name} to {output_dir}")
            
        except ImportError:
            logger.error("Kaggle API not installed. Install with: pip install kaggle")
        except Exception as e:
            logger.error(f"Error downloading Kaggle dataset: {str(e)}")
    
    def create_dataset_info(self):
        """Create dataset information file."""
        dataset_info = {
            "datasets": {
                "BraTS": {
                    "description": "Brain Tumor Segmentation Challenge",
                    "url": "https://www.med.upenn.edu/cbica/brats2020/",
                    "classes": {
                        "0": "Background",
                        "1": "Necrotic and non-enhancing tumor core",
                        "2": "Peritumoral edema",
                        "3": "GD-enhancing tumor"
                    },
                    "modalities": ["T1", "T1ce", "T2", "FLAIR"]
                },
                "TCIA": {
                    "description": "The Cancer Imaging Archive",
                    "url": "https://www.cancerimagingarchive.net/",
                    "note": "Requires manual download and account creation"
                },
                "Kaggle_Brain_Tumor": {
                    "description": "Brain Tumor Classification Dataset",
                    "url": "https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri",
                    "classes": ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
                }
            },
            "directory_structure": {
                "raw/": "Original downloaded datasets",
                "processed/": "Preprocessed and normalized data",
                "samples/": "Sample data for testing",
                "models/": "Trained model files"
            }
        }
        
        info_file = self.data_dir / "dataset_info.json"
        import json
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Dataset information saved to {info_file}")
    
    def list_available_datasets(self):
        """List information about available datasets."""
        logger.info("\n" + "="*50)
        logger.info("AVAILABLE BRAIN TUMOR DATASETS")
        logger.info("="*50)
        
        datasets = [
            {
                "name": "BraTS 2020",
                "description": "Multimodal Brain Tumor Segmentation Challenge",
                "size": "~7GB",
                "samples": "369 training, 125 validation",
                "url": "https://www.med.upenn.edu/cbica/brats2020/"
            },
            {
                "name": "TCIA Brain Tumor",
                "description": "The Cancer Imaging Archive brain tumor collections",
                "size": "Varies",
                "samples": "1000+ studies",
                "url": "https://www.cancerimagingarchive.net/"
            },
            {
                "name": "Kaggle Brain Tumor MRI",
                "description": "Brain Tumor Classification Dataset",
                "size": "~50MB",
                "samples": "3264 images",
                "url": "https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri"
            }
        ]
        
        for dataset in datasets:
            logger.info(f"\nDataset: {dataset['name']}")
            logger.info(f"  Description: {dataset['description']}")
            logger.info(f"  Size: {dataset['size']}")
            logger.info(f"  Samples: {dataset['samples']}")
            logger.info(f"  URL: {dataset['url']}")


def main():
    """Main function for data download script."""
    parser = argparse.ArgumentParser(description="Brain Tumor Dataset Downloader")
    parser.add_argument("--data-dir", default="data/raw", help="Data directory")
    parser.add_argument("--dataset", choices=["sample", "brats", "kaggle", "all"], 
                       default="sample", help="Dataset to download")
    parser.add_argument("--kaggle-dataset", help="Specific Kaggle dataset name")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    
    args = parser.parse_args()
    
    downloader = DataDownloader(args.data_dir)
    
    if args.list:
        downloader.list_available_datasets()
        return
    
    if args.dataset == "sample" or args.dataset == "all":
        downloader.download_sample_data()
    
    if args.dataset == "brats" or args.dataset == "all":
        downloader.setup_brats_data_structure()
    
    if args.dataset == "kaggle" and args.kaggle_dataset:
        downloader.download_kaggle_dataset(args.kaggle_dataset)
    
    # Always create dataset info
    downloader.create_dataset_info()
    
    logger.info("Data download script completed!")


if __name__ == "__main__":
    main()