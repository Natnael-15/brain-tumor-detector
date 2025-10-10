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

# Try to import medical imaging libraries
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

logger = logging.getLogger(__name__)

class MedicalVisualizationService:
    """Service for 3D medical image visualization and processing"""
    
    def __init__(self):
        self.supported_formats = ['.nii', '.nii.gz', '.dcm', '.mha', '.mhd']
        self.visualization_cache = {}
        
    async def generate_3d_visualization(self, file_path: str, analysis_id: str) -> Dict[str, Any]:
        """Generate 3D visualization data for medical images"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Medical image file not found: {file_path}")
            
            # Check if already cached
            cache_key = f"{file_path.name}_{file_path.stat().st_mtime}"
            if cache_key in self.visualization_cache:
                logger.info(f"Using cached visualization for {file_path.name}")
                return self.visualization_cache[cache_key]
            
            # Load medical image
            image_data, metadata = await self._load_medical_image(file_path)
            
            # Generate different visualization types
            visualization_data = {
                "analysis_id": analysis_id,
                "file_name": file_path.name,
                "generated_at": datetime.now().isoformat(),
                "metadata": metadata,
                "visualizations": {
                    "volume_rendering": await self._generate_volume_rendering(image_data, metadata),
                    "slice_views": await self._generate_slice_views(image_data, metadata),
                    "mpr_views": await self._generate_mpr_views(image_data, metadata),
                    "histogram": await self._generate_intensity_histogram(image_data),
                    "statistics": await self._calculate_image_statistics(image_data)
                },
                "interaction_data": {
                    "volume_bounds": self._get_volume_bounds(image_data, metadata),
                    "slice_ranges": self._get_slice_ranges(image_data),
                    "intensity_range": self._get_intensity_range(image_data),
                    "voxel_spacing": metadata.get("spacing", [1.0, 1.0, 1.0])
                }
            }
            
            # Cache the result
            self.visualization_cache[cache_key] = visualization_data
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"Error generating 3D visualization: {e}")
            # Return mock visualization for development
            return await self._generate_mock_visualization(str(file_path), analysis_id)\n    \n    async def _load_medical_image(self, file_path: Path) -> Tuple[np.ndarray, Dict]:\n        \"\"\"Load medical image using appropriate library\"\"\"\n        try:\n            if NIBABEL_AVAILABLE and file_path.suffix.lower() in ['.nii', '.gz']:\n                # Load with nibabel\n                img = nib.load(str(file_path))\n                data = img.get_fdata()\n                \n                metadata = {\n                    \"shape\": data.shape,\n                    \"spacing\": img.header.get_zooms()[:3] if hasattr(img.header, 'get_zooms') else [1.0, 1.0, 1.0],\n                    \"origin\": [0.0, 0.0, 0.0],\n                    \"orientation\": \"RAS\",\n                    \"data_type\": str(data.dtype),\n                    \"file_format\": \"NIfTI\"\n                }\n                \n                return data, metadata\n                \n            elif SITK_AVAILABLE:\n                # Load with SimpleITK\n                img = sitk.ReadImage(str(file_path))\n                data = sitk.GetArrayFromImage(img)\n                \n                metadata = {\n                    \"shape\": data.shape,\n                    \"spacing\": img.GetSpacing(),\n                    \"origin\": img.GetOrigin(),\n                    \"direction\": img.GetDirection(),\n                    \"data_type\": str(data.dtype),\n                    \"file_format\": \"SimpleITK\"\n                }\n                \n                return data, metadata\n            else:\n                # Generate mock data for development\n                logger.warning(f\"Medical imaging libraries not available, generating mock data\")\n                return await self._generate_mock_image_data()\n                \n        except Exception as e:\n            logger.error(f\"Error loading medical image {file_path}: {e}\")\n            return await self._generate_mock_image_data()\n    \n    async def _generate_mock_image_data(self) -> Tuple[np.ndarray, Dict]:\n        \"\"\"Generate mock 3D medical image data for development\"\"\"\n        # Create a 3D volume with some structure\n        shape = (128, 128, 64)\n        data = np.random.randn(*shape) * 50 + 100\n        \n        # Add some \"tumor-like\" structures\n        center = (64, 64, 32)\n        x, y, z = np.meshgrid(\n            np.arange(shape[0]) - center[0],\n            np.arange(shape[1]) - center[1], \n            np.arange(shape[2]) - center[2],\n            indexing='ij'\n        )\n        \n        # Add spherical \"tumor\"\n        radius = 15\n        tumor_mask = (x**2 + y**2 + z**2) <= radius**2\n        data[tumor_mask] = 200\n        \n        # Add some noise\n        data += np.random.normal(0, 10, shape)\n        data = np.clip(data, 0, 255).astype(np.uint8)\n        \n        metadata = {\n            \"shape\": shape,\n            \"spacing\": [1.0, 1.0, 1.0],\n            \"origin\": [0.0, 0.0, 0.0],\n            \"orientation\": \"RAS\",\n            \"data_type\": \"uint8\",\n            \"file_format\": \"Mock\"\n        }\n        \n        return data, metadata\n    \n    async def _generate_volume_rendering(self, data: np.ndarray, metadata: Dict) -> Dict:\n        \"\"\"Generate volume rendering data\"\"\"\n        # Downsample for web viewing\n        downsampled = data[::2, ::2, ::2] if data.size > 1000000 else data\n        \n        # Generate transfer function\n        min_val, max_val = np.min(data), np.max(data)\n        \n        return {\n            \"type\": \"volume_rendering\",\n            \"dimensions\": list(downsampled.shape),\n            \"intensity_range\": [float(min_val), float(max_val)],\n            \"transfer_function\": {\n                \"opacity\": [\n                    {\"value\": float(min_val), \"opacity\": 0.0},\n                    {\"value\": float(min_val + (max_val - min_val) * 0.3), \"opacity\": 0.1},\n                    {\"value\": float(min_val + (max_val - min_val) * 0.7), \"opacity\": 0.8},\n                    {\"value\": float(max_val), \"opacity\": 1.0}\n                ],\n                \"color\": [\n                    {\"value\": float(min_val), \"color\": [0, 0, 0]},\n                    {\"value\": float(max_val * 0.5), \"color\": [128, 128, 128]},\n                    {\"value\": float(max_val), \"color\": [255, 255, 255]}\n                ]\n            },\n            \"spacing\": metadata.get(\"spacing\", [1.0, 1.0, 1.0]),\n            \"data_url\": f\"/api/v1/visualization/volume_data/{id(data)}\"\n        }\n    \n    async def _generate_slice_views(self, data: np.ndarray, metadata: Dict) -> Dict:\n        \"\"\"Generate axial, sagittal, and coronal slice views\"\"\"\n        shape = data.shape\n        \n        # Get middle slices\n        axial_slice = shape[2] // 2\n        sagittal_slice = shape[0] // 2\n        coronal_slice = shape[1] // 2\n        \n        return {\n            \"type\": \"slice_views\",\n            \"views\": {\n                \"axial\": {\n                    \"slice_index\": axial_slice,\n                    \"max_slices\": shape[2],\n                    \"orientation\": \"axial\",\n                    \"image_url\": f\"/api/v1/visualization/slice/axial/{id(data)}/{axial_slice}\"\n                },\n                \"sagittal\": {\n                    \"slice_index\": sagittal_slice,\n                    \"max_slices\": shape[0],\n                    \"orientation\": \"sagittal\",\n                    \"image_url\": f\"/api/v1/visualization/slice/sagittal/{id(data)}/{sagittal_slice}\"\n                },\n                \"coronal\": {\n                    \"slice_index\": coronal_slice,\n                    \"max_slices\": shape[1],\n                    \"orientation\": \"coronal\",\n                    \"image_url\": f\"/api/v1/visualization/slice/coronal/{id(data)}/{coronal_slice}\"\n                }\n            },\n            \"window_level\": {\n                \"min\": float(np.min(data)),\n                \"max\": float(np.max(data)),\n                \"default_window\": float(np.max(data) - np.min(data)),\n                \"default_level\": float(np.mean(data))\n            }\n        }\n    \n    async def _generate_mpr_views(self, data: np.ndarray, metadata: Dict) -> Dict:\n        \"\"\"Generate Multi-Planar Reconstruction views\"\"\"\n        return {\n            \"type\": \"mpr_views\",\n            \"description\": \"Multi-Planar Reconstruction with cross-hair navigation\",\n            \"features\": [\n                \"Synchronized cross-hair cursor\",\n                \"Real-time slice updates\",\n                \"Window/Level adjustment\",\n                \"Zoom and pan controls\"\n            ],\n            \"viewport_config\": {\n                \"layout\": \"2x2\",\n                \"views\": [\"axial\", \"sagittal\", \"coronal\", \"3d\"]\n            },\n            \"interaction_url\": f\"/api/v1/visualization/mpr/{id(data)}\"\n        }\n    \n    async def _generate_intensity_histogram(self, data: np.ndarray) -> Dict:\n        \"\"\"Generate intensity histogram\"\"\"\n        hist, bins = np.histogram(data.flatten(), bins=100)\n        \n        return {\n            \"type\": \"histogram\",\n            \"bins\": bins[:-1].tolist(),\n            \"counts\": hist.tolist(),\n            \"statistics\": {\n                \"mean\": float(np.mean(data)),\n                \"std\": float(np.std(data)),\n                \"min\": float(np.min(data)),\n                \"max\": float(np.max(data)),\n                \"median\": float(np.median(data))\n            }\n        }\n    \n    async def _calculate_image_statistics(self, data: np.ndarray) -> Dict:\n        \"\"\"Calculate comprehensive image statistics\"\"\"\n        return {\n            \"voxel_count\": int(data.size),\n            \"non_zero_voxels\": int(np.count_nonzero(data)),\n            \"mean_intensity\": float(np.mean(data)),\n            \"std_intensity\": float(np.std(data)),\n            \"min_intensity\": float(np.min(data)),\n            \"max_intensity\": float(np.max(data)),\n            \"median_intensity\": float(np.median(data)),\n            \"percentiles\": {\n                \"p10\": float(np.percentile(data, 10)),\n                \"p25\": float(np.percentile(data, 25)),\n                \"p75\": float(np.percentile(data, 75)),\n                \"p90\": float(np.percentile(data, 90)),\n                \"p95\": float(np.percentile(data, 95)),\n                \"p99\": float(np.percentile(data, 99))\n            }\n        }\n    \n    def _get_volume_bounds(self, data: np.ndarray, metadata: Dict) -> List[float]:\n        \"\"\"Get physical volume bounds\"\"\"\n        shape = data.shape\n        spacing = metadata.get(\"spacing\", [1.0, 1.0, 1.0])\n        origin = metadata.get(\"origin\", [0.0, 0.0, 0.0])\n        \n        return [\n            origin[0], origin[0] + shape[0] * spacing[0],\n            origin[1], origin[1] + shape[1] * spacing[1],\n            origin[2], origin[2] + shape[2] * spacing[2]\n        ]\n    \n    def _get_slice_ranges(self, data: np.ndarray) -> Dict[str, int]:\n        \"\"\"Get slice ranges for each orientation\"\"\"\n        return {\n            \"axial\": data.shape[2],\n            \"sagittal\": data.shape[0],\n            \"coronal\": data.shape[1]\n        }\n    \n    def _get_intensity_range(self, data: np.ndarray) -> Tuple[float, float]:\n        \"\"\"Get intensity range\"\"\"\n        return float(np.min(data)), float(np.max(data))\n    \n    async def _generate_mock_visualization(self, file_path: str, analysis_id: str) -> Dict[str, Any]:\n        \"\"\"Generate mock visualization for development\"\"\"\n        return {\n            \"analysis_id\": analysis_id,\n            \"file_name\": Path(file_path).name,\n            \"generated_at\": datetime.now().isoformat(),\n            \"metadata\": {\n                \"shape\": [128, 128, 64],\n                \"spacing\": [1.0, 1.0, 1.0],\n                \"data_type\": \"mock\"\n            },\n            \"visualizations\": {\n                \"volume_rendering\": {\n                    \"type\": \"volume_rendering\",\n                    \"available\": True,\n                    \"features\": [\"3D volume rendering\", \"Transfer function editor\", \"Lighting controls\"]\n                },\n                \"slice_views\": {\n                    \"type\": \"slice_views\",\n                    \"views\": [\"axial\", \"sagittal\", \"coronal\"],\n                    \"interactive\": True\n                },\n                \"mpr_views\": {\n                    \"type\": \"mpr_views\",\n                    \"description\": \"Multi-Planar Reconstruction ready\"\n                }\n            },\n            \"status\": \"mock_data\",\n            \"message\": \"3D visualization service ready - connect medical imaging libraries for full functionality\"\n        }\n    \n    async def generate_tumor_overlay(self, analysis_id: str, segmentation_data: Dict) -> Dict[str, Any]:\n        \"\"\"Generate tumor overlay for 3D visualization\"\"\"\n        return {\n            \"analysis_id\": analysis_id,\n            \"overlay_type\": \"tumor_segmentation\",\n            \"generated_at\": datetime.now().isoformat(),\n            \"overlay_data\": {\n                \"tumor_regions\": [\n                    {\n                        \"id\": \"tumor_1\",\n                        \"type\": \"enhancing_tumor\",\n                        \"color\": [255, 0, 0, 128],\n                        \"volume_ml\": segmentation_data.get(\"prediction\", {}).get(\"volume_ml\", 0),\n                        \"confidence\": segmentation_data.get(\"prediction\", {}).get(\"confidence\", 0)\n                    },\n                    {\n                        \"id\": \"edema\",\n                        \"type\": \"peritumoral_edema\",\n                        \"color\": [0, 255, 0, 64],\n                        \"volume_ml\": 0,\n                        \"confidence\": 0.8\n                    }\n                ],\n                \"visualization_settings\": {\n                    \"opacity\": 0.7,\n                    \"blend_mode\": \"multiply\",\n                    \"show_contours\": True,\n                    \"contour_thickness\": 1.0\n                }\n            }\n        }\n    \n    def clear_cache(self):\n        \"\"\"Clear visualization cache\"\"\"\n        self.visualization_cache.clear()\n        logger.info(\"Visualization cache cleared\")\n\n\n# Global visualization service instance\nvisualization_service = MedicalVisualizationService()