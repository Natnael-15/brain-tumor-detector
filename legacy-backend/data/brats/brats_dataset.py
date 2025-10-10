#!/usr/bin/env python3
"""
BraTS Dataset Handler for Brain MRI Tumor Detection.

Implements dataset loading, preprocessing, and evaluation utilities
for the Brain Tumor Segmentation (BraTS) challenge dataset.
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import logging
from abc import ABC, abstractmethod

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logging.warning("nibabel not available. Install with: pip install nibabel")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available for dataset loading")


class BraTSDataset(Dataset if TORCH_AVAILABLE else object):
    """
    BraTS Dataset class for loading and processing BraTS data.
    
    Handles multimodal MRI data (T1, T1ce, T2, FLAIR) and segmentation masks
    from the Brain Tumor Segmentation challenge dataset.
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 modalities: List[str] = ['t1', 't1ce', 't2', 'flair'],
                 transform: Optional[Callable] = None,
                 load_seg: bool = True,
                 cache_data: bool = False,
                 preprocessing: Optional[Callable] = None):
        """
        Initialize BraTS dataset.
        
        Args:
            data_dir: Path to BraTS dataset directory
            split: Dataset split ('train', 'val', 'test')
            modalities: List of modalities to load
            transform: Optional transform function
            load_seg: Whether to load segmentation masks
            cache_data: Whether to cache loaded data in memory
            preprocessing: Optional preprocessing function
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.modalities = modalities
        self.transform = transform
        self.load_seg = load_seg
        self.cache_data = cache_data
        self.preprocessing = preprocessing
        
        self.logger = logging.getLogger(__name__)
        
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required for BraTS dataset. Install with: pip install nibabel")
            
        # Find data files
        self.data_list = self._find_data_files()
        
        # Cache for loaded data
        self.data_cache = {} if cache_data else None
        
        self.logger.info(f"Initialized BraTS dataset with {len(self.data_list)} samples")
        
    def _find_data_files(self) -> List[Dict[str, str]]:
        """
        Find and organize BraTS data files.
        
        Returns:
            List of dictionaries containing file paths for each subject
        """
        data_list = []
        
        # BraTS data structure: BraTS_2021_TrainingData/BraTS2021_xxxxx/
        pattern = str(self.data_dir / "**" / "BraTS*")
        subject_dirs = glob.glob(pattern)
        
        for subject_dir in sorted(subject_dirs):
            subject_path = Path(subject_dir)
            subject_id = subject_path.name
            
            # Find modality files
            files = {}
            for modality in self.modalities:
                pattern = f"{subject_id}_{modality}.nii.gz"
                file_path = subject_path / pattern
                
                if file_path.exists():
                    files[modality] = str(file_path)
                else:
                    self.logger.warning(f"Missing {modality} for {subject_id}")
                    
            # Find segmentation file
            if self.load_seg:
                seg_pattern = f"{subject_id}_seg.nii.gz"
                seg_path = subject_path / seg_pattern
                
                if seg_path.exists():
                    files['seg'] = str(seg_path)
                elif self.split == 'train':  # Training data should have segmentations
                    self.logger.warning(f"Missing segmentation for {subject_id}")
                    
            if len(files) >= len(self.modalities):  # Has all required modalities
                files['subject_id'] = subject_id
                files['subject_dir'] = str(subject_path)
                data_list.append(files)
                
        return data_list
        
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data_list)
        
    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, str]]:
        """
        Get dataset item.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing loaded data
        """
        if self.cache_data and idx in self.data_cache:
            return self.data_cache[idx]
            
        data_info = self.data_list[idx]
        sample = {'subject_id': data_info['subject_id']}
        
        # Load modality images
        images = []
        for modality in self.modalities:
            if modality in data_info:
                img_path = data_info[modality]
                img = self._load_nifti(img_path)
                images.append(img)
            else:
                # Create dummy image if modality missing
                images.append(np.zeros((240, 240, 155), dtype=np.float32))
                
        # Stack modalities
        if images:
            sample['image'] = np.stack(images, axis=0)  # (C, H, W, D)
        else:
            sample['image'] = np.zeros((len(self.modalities), 240, 240, 155), dtype=np.float32)
            
        # Load segmentation if available
        if self.load_seg and 'seg' in data_info:
            seg_path = data_info['seg']
            sample['seg'] = self._load_nifti(seg_path, is_seg=True)
        elif self.load_seg:
            sample['seg'] = np.zeros((240, 240, 155), dtype=np.int32)
            
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(sample)
            
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
            
        # Cache data if enabled
        if self.cache_data:
            self.data_cache[idx] = sample
            
        return sample
        
    def _load_nifti(self, file_path: str, is_seg: bool = False) -> np.ndarray:
        """
        Load NIfTI file.
        
        Args:
            file_path: Path to NIfTI file
            is_seg: Whether this is a segmentation file
            
        Returns:
            Loaded image array
        """
        try:
            nii = nib.load(file_path)
            data = nii.get_fdata()
            
            if is_seg:
                data = data.astype(np.int32)
            else:
                data = data.astype(np.float32)
                
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            # Return dummy data
            if is_seg:
                return np.zeros((240, 240, 155), dtype=np.int32)
            else:
                return np.zeros((240, 240, 155), dtype=np.float32)
                
    def get_subject_info(self, idx: int) -> Dict[str, str]:
        """
        Get information about a specific subject.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with subject information
        """
        return self.data_list[idx]
        
    def get_statistics(self) -> Dict[str, float]:
        """
        Compute dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_subjects': len(self.data_list),
            'num_modalities': len(self.modalities),
            'has_segmentations': sum(1 for item in self.data_list if 'seg' in item)
        }
        
        # Compute image statistics (on subset for efficiency)
        subset_size = min(10, len(self.data_list))
        intensities = []
        
        for i in range(subset_size):
            sample = self[i]
            image = sample['image']
            # Remove background (intensity = 0)
            foreground = image[image > 0]
            if len(foreground) > 0:
                intensities.extend(foreground.flatten())
                
        if intensities:
            intensities = np.array(intensities)
            stats.update({
                'mean_intensity': float(np.mean(intensities)),
                'std_intensity': float(np.std(intensities)),
                'min_intensity': float(np.min(intensities)),
                'max_intensity': float(np.max(intensities))
            })
            
        return stats


class BraTSDataLoader:
    """
    Data loader for BraTS dataset with batch processing.
    """
    
    def __init__(self,
                 dataset: BraTSDataset,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True):
        """
        Initialize BraTS data loader.
        
        Args:
            dataset: BraTSDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        if TORCH_AVAILABLE:
            self.dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=self._collate_fn
            )
        else:
            raise ImportError("PyTorch is required for BraTSDataLoader")
            
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching BraTS data.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched data dictionary
        """
        # Stack images
        images = torch.stack([torch.from_numpy(item['image']) for item in batch])
        
        result = {
            'image': images,
            'subject_id': [item['subject_id'] for item in batch]
        }
        
        # Stack segmentations if available
        if 'seg' in batch[0]:
            segs = torch.stack([torch.from_numpy(item['seg']) for item in batch])
            result['seg'] = segs
            
        return result
        
    def __iter__(self):
        """Iterate over data loader."""
        return iter(self.dataloader)
        
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.dataloader)


class BraTSSubsetDataset(BraTSDataset):
    """
    Subset of BraTS dataset for development and testing.
    """
    
    def __init__(self, dataset: BraTSDataset, indices: List[int]):
        """
        Initialize subset dataset.
        
        Args:
            dataset: Parent BraTSDataset
            indices: List of indices to include in subset
        """
        self.parent_dataset = dataset
        self.indices = indices
        
        # Copy relevant attributes
        self.data_dir = dataset.data_dir
        self.split = dataset.split
        self.modalities = dataset.modalities
        self.transform = dataset.transform
        self.load_seg = dataset.load_seg
        self.preprocessing = dataset.preprocessing
        self.logger = dataset.logger
        
    def __len__(self) -> int:
        return len(self.indices)
        
    def __getitem__(self, idx: int):
        return self.parent_dataset[self.indices[idx]]
        
    def get_subject_info(self, idx: int) -> Dict[str, str]:
        return self.parent_dataset.get_subject_info(self.indices[idx])