# 3D Visualization Service for Medical Imaging
# Phase 3 Step 2: Advanced 3D Medical Image Viewing

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
import json
import base64
from pathlib import Path
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class MedicalVisualizationService:
    """Service for 3D medical image visualization and processing"""
    
    def __init__(self):
        self.supported_3d_formats = ['.nii', '.nii.gz', '.dcm', '.mha', '.mhd']
        self.supported_2d_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.visualization_cache = {}
        
    async def generate_3d_visualization(self, file_path: str, analysis_id: str) -> Dict[str, Any]:
        """Generate 3D visualization data for medical images"""
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                raise FileNotFoundError(f"Medical image file not found: {file_path}")
            
            # Check file type
            file_extension = file_path_obj.suffix.lower()
            
            if file_extension in self.supported_2d_formats:
                # Handle 2D image files (JPG, PNG, etc.)
                return await self._generate_2d_visualization(file_path, analysis_id)
            else:
                # Handle 3D medical files or fallback to mock
                return await self._generate_mock_3d_visualization(file_path, analysis_id)
            
        except Exception as e:
            logger.error(f"Error generating 3D visualization: {e}")
            # Return mock visualization for development
            return await self._generate_mock_3d_visualization(file_path, analysis_id)
    
    async def _generate_2d_visualization(self, file_path: str, analysis_id: str) -> Dict[str, Any]:
        """Generate visualization for 2D image files (JPG, PNG, etc.)"""
        return {
            "analysis_id": analysis_id,
            "file_name": Path(file_path).name,
            "generated_at": datetime.now().isoformat(),
            "is_2d_image": True,
            "metadata": {
                "dimensions": [512, 512, 1],  # Treat as single slice
                "spacing": [0.5, 0.5, 1.0],  # Higher resolution for 2D
                "origin": [-256.0, -256.0, 0.0],
                "orientation": "RAS",
                "modality": "2D Image",
                "data_type": "2d_slice",
                "message": "2D image file detected. Displaying as single slice."
            },
            "volume_data": self._generate_single_slice_data(),
            "tumor_overlay": None,  # No tumor overlay for 2D images
            "visualizations": {
                "slice_views": {
                    "type": "single_slice",
                    "views": {
                        "axial": {
                            "slice_index": 0,
                            "max_slices": 1,
                            "orientation": "axial",
                            "image_url": f"/api/v1/visualization/slice/axial/{analysis_id}/0"
                        }
                    },
                    "note": "2D Image Visualization",
                    "message": "This is a 2D image file (JPG/PNG). The analysis has been performed, but 3D visualization is limited. For comprehensive tumor detection and 3D visualization, please upload DICOM (.dcm) or NIfTI (.nii) medical imaging files.",
                    "recommendations": [
                        "Upload DICOM files for medical-grade 3D visualization",
                        "Use NIfTI format for research and advanced analysis", 
                        "Consider MRI or CT scan data for accurate tumor detection",
                        "2D images provide limited clinical information"
                    ]
                }
            }
        }
    
    async def _generate_mock_3d_visualization(self, file_path: str, analysis_id: str) -> Dict[str, Any]:
        """Generate mock 3D visualization for development"""
        return {
            "analysis_id": analysis_id,
            "file_name": Path(file_path).name,
            "generated_at": datetime.now().isoformat(),
            "metadata": {
                # Realistic brain MRI dimensions (typical T1-weighted scan)
                "dimensions": [256, 256, 180],  # Width x Height x Depth (voxels)
                "spacing": [1.0, 1.0, 1.0],    # 1mm isotropic voxels (medical standard)
                "origin": [-128.0, -128.0, -90.0],  # Center the volume
                "orientation": "RAS",  # Right-Anterior-Superior (medical standard)
                "modality": "T1-weighted MRI",
                "field_of_view": [256.0, 256.0, 180.0],  # FOV in mm
                "data_type": "mock_medical"
            },
            "volume_data": self._generate_mock_volume_data(),
            "tumor_overlay": self._generate_mock_tumor_data(),
            "visualizations": {
                "volume_rendering": {
                    "type": "volume_rendering",
                    "available": True,
                    "features": ["3D volume rendering", "Transfer function editor", "Lighting controls"],
                    "dimensions": [256, 256, 180],
                    "intensity_range": [0, 4095],  # 12-bit medical imaging range
                    "data_url": f"/api/v1/visualization/volume_data/{analysis_id}"
                },
                "slice_views": {
                    "type": "slice_views",
                    "views": {
                        "axial": {
                            "slice_index": 90,   # Middle slice
                            "max_slices": 180,   # Total depth
                            "orientation": "axial",
                            "image_url": f"/api/v1/visualization/slice/axial/{analysis_id}/90"
                        },
                        "sagittal": {
                            "slice_index": 128,  # Middle slice
                            "max_slices": 256,   # Total width
                            "orientation": "sagittal", 
                            "image_url": f"/api/v1/visualization/slice/sagittal/{analysis_id}/128"
                        },
                        "coronal": {
                            "slice_index": 128,  # Middle slice
                            "max_slices": 256,   # Total height
                            "orientation": "coronal",
                            "image_url": f"/api/v1/visualization/slice/coronal/{analysis_id}/128"
                        }
                    },
                    "window_level": {
                        "min": 0,
                        "max": 4095,        # 12-bit medical range
                        "default_window": 2000,  # Typical brain window
                        "default_level": 1000    # Typical brain level
                    }
                },
                "mpr_views": {
                    "type": "mpr_views",
                    "description": "Multi-Planar Reconstruction with cross-hair navigation",
                    "features": [
                        "Synchronized cross-hair cursor",
                        "Real-time slice updates",
                        "Window/Level adjustment",
                        "Zoom and pan controls"
                    ],
                    "viewport_config": {
                        "layout": "2x2",
                        "views": ["axial", "sagittal", "coronal", "3d"]
                    },
                    "interaction_url": f"/api/v1/visualization/mpr/{analysis_id}"
                },
                "histogram": {
                    "type": "histogram",
                    "bins": list(range(0, 4096, 100)),  # 12-bit range with 100-unit bins
                    "counts": [np.random.randint(100, 2000) for _ in range(41)],
                    "statistics": {
                        "mean": 1024.5,  # Typical brain tissue intensity
                        "std": 345.2,
                        "min": 0,
                        "max": 4095,
                        "median": 950
                    }
                }
            },
            "interaction_data": {
                "volume_bounds": [-128.0, 128.0, -128.0, 128.0, -90.0, 90.0],  # Real-world coordinates in mm
                "slice_ranges": {"axial": 180, "sagittal": 256, "coronal": 256},
                "intensity_range": [0, 4095],  # 12-bit medical range
                "voxel_spacing": [1.0, 1.0, 1.0],  # 1mm isotropic
                "coordinate_system": "RAS"  # Right-Anterior-Superior
            },
            "status": "mock_data",
            "message": "3D visualization service ready - Phase 3 Step 2 implementation"
        }
    
    async def generate_tumor_overlay(self, analysis_id: str, segmentation_data: Dict) -> Dict[str, Any]:
        """Generate tumor overlay for 3D visualization"""
        return {
            "analysis_id": analysis_id,
            "overlay_type": "tumor_segmentation",
            "generated_at": datetime.now().isoformat(),
            "overlay_data": {
                "tumor_regions": [
                    {
                        "id": "tumor_1",
                        "type": "enhancing_tumor",
                        "color": [255, 0, 0, 128],
                        "volume_ml": segmentation_data.get("prediction", {}).get("volume_ml", 0),
                        "confidence": segmentation_data.get("prediction", {}).get("confidence", 0)
                    },
                    {
                        "id": "edema",
                        "type": "peritumoral_edema", 
                        "color": [0, 255, 0, 64],
                        "volume_ml": 0,
                        "confidence": 0.8
                    }
                ],
                "visualization_settings": {
                    "opacity": 0.7,
                    "blend_mode": "multiply",
                    "show_contours": True,
                    "contour_thickness": 1.0
                }
            }
        }
    
    def _generate_single_slice_data(self) -> List[List[List[float]]]:
        """Generate mock data for a single 2D slice"""
        # Generate a single slice with more interesting pattern for 2D images
        slice_data = []
        size = 128  # Smaller size for 2D display
        
        for i in range(size):
            row = []
            for j in range(size):
                # Create a more interesting 2D pattern
                center_x, center_y = size // 2, size // 2
                distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                
                # Create a brain-like circular pattern
                if distance < size * 0.4:
                    # Inside "brain" area
                    value = 0.7 + 0.3 * np.sin(distance * 0.2) * np.cos(i * 0.1) * np.cos(j * 0.1)
                    # Add some "tumor-like" bright spots
                    if (distance > size * 0.15 and distance < size * 0.25 and 
                        abs(i - center_x) < 10 and abs(j - center_y + 15) < 8):
                        value = min(1.0, value + 0.4)  # Bright spot
                else:
                    # Outside brain area
                    value = 0.1 + 0.1 * np.random.random()
                
                row.append(float(np.clip(value, 0, 1)))
            slice_data.append(row)
        
        return [slice_data]  # Return as single slice in volume

    def _generate_mock_volume_data(self) -> List[List[List[int]]]:
        """Generate mock 3D volume data for visualization"""
        # Create a simple 3D brain-like structure
        depth, height, width = 32, 32, 32  # Smaller for demo
        volume = []
        
        for z in range(depth):
            slice_data = []
            for y in range(height):
                row = []
                for x in range(width):
                    # Create brain-like intensity pattern
                    center_x, center_y, center_z = width//2, height//2, depth//2
                    dist_from_center = ((x-center_x)**2 + (y-center_y)**2 + (z-center_z)**2)**0.5
                    
                    if dist_from_center < 12:  # Brain tissue
                        intensity = int(200 + np.random.normal(0, 20))
                    elif dist_from_center < 15:  # Skull
                        intensity = int(150 + np.random.normal(0, 10))
                    else:  # Background
                        intensity = int(50 + np.random.normal(0, 5))
                    
                    row.append(max(0, min(255, intensity)))
                slice_data.append(row)
            volume.append(slice_data)
        
        return volume
    
    def _generate_mock_tumor_data(self) -> List[List[List[int]]]:
        """Generate mock tumor overlay data"""
        # Create a simple tumor region
        depth, height, width = 32, 32, 32
        tumor = []
        
        for z in range(depth):
            slice_data = []
            for y in range(height):
                row = []
                for x in range(width):
                    # Create tumor in upper right quadrant
                    if (x > 20 and x < 28 and y > 20 and y < 28 and z > 16 and z < 24):
                        row.append(255)  # Tumor region
                    else:
                        row.append(0)    # No tumor
                slice_data.append(row)
            tumor.append(slice_data)
        
        return tumor

    def clear_cache(self):
        """Clear visualization cache"""
        self.visualization_cache.clear()
        logger.info("Visualization cache cleared")


# Global visualization service instance
visualization_service = MedicalVisualizationService()