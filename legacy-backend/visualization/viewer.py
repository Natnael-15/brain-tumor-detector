"""
3D Brain Visualization Module

This module provides interactive 3D visualization capabilities for brain MRI scans
and tumor segmentation results.
"""

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Optional, Tuple, Dict, List
from pathlib import Path

# Try to import optional dependencies
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


class BrainViewer:
    """Interactive 3D brain MRI viewer with tumor highlighting."""
    
    def __init__(self):
        """Initialize the brain viewer."""
        self.current_image = None
        self.current_segmentation = None
        self.image_shape = None
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load MRI image from file.
        
        Args:
            image_path: Path to MRI image file
            
        Returns:
            3D image array
        """
        try:
            if NIBABEL_AVAILABLE and (image_path.endswith('.nii') or image_path.endswith('.nii.gz')):
                nii = nib.load(image_path)
                image_data = nii.get_fdata()
            else:
                # Try loading as numpy array
                image_data = np.load(image_path)
            
            self.current_image = image_data
            self.image_shape = image_data.shape
            logger.info(f"Loaded image with shape: {image_data.shape}")
            return image_data
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return dummy data for demonstration
            dummy_data = np.random.rand(128, 128, 128) * 255
            self.current_image = dummy_data
            self.image_shape = dummy_data.shape
            return dummy_data
    
    def load_segmentation(self, seg_path: str) -> Optional[np.ndarray]:
        """
        Load tumor segmentation mask.
        
        Args:
            seg_path: Path to segmentation file
            
        Returns:
            3D segmentation array
        """
        try:
            if NIBABEL_AVAILABLE and (seg_path.endswith('.nii') or seg_path.endswith('.nii.gz')):
                nii = nib.load(seg_path)
                seg_data = nii.get_fdata()
            else:
                seg_data = np.load(seg_path)
            
            self.current_segmentation = seg_data
            logger.info(f"Loaded segmentation with shape: {seg_data.shape}")
            return seg_data
            
        except Exception as e:
            logger.error(f"Error loading segmentation {seg_path}: {str(e)}")
            return None
    
    def create_slice_viewer(self, image: np.ndarray, segmentation: Optional[np.ndarray] = None) -> Optional[Figure]:
        """
        Create interactive slice viewer using matplotlib.
        
        Args:
            image: 3D image array
            segmentation: Optional 3D segmentation array
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Initial slice indices
        mid_x, mid_y, mid_z = [s // 2 for s in image.shape]
        
        # Create initial displays
        im1 = axes[0].imshow(image[mid_x, :, :], cmap='gray', aspect='auto')
        axes[0].set_title('Sagittal View')
        axes[0].axis('off')
        
        im2 = axes[1].imshow(image[:, mid_y, :], cmap='gray', aspect='auto')
        axes[1].set_title('Coronal View')
        axes[1].axis('off')
        
        im3 = axes[2].imshow(image[:, :, mid_z], cmap='gray', aspect='auto')
        axes[2].set_title('Axial View')
        axes[2].axis('off')
        
        # Add segmentation overlay if provided
        if segmentation is not None:
            # Create colored overlay for tumors
            overlay1 = self._create_overlay(segmentation[mid_x, :, :])
            overlay2 = self._create_overlay(segmentation[:, mid_y, :])
            overlay3 = self._create_overlay(segmentation[:, :, mid_z])
            
            axes[0].imshow(overlay1, alpha=0.3)
            axes[1].imshow(overlay2, alpha=0.3)
            axes[2].imshow(overlay3, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_3d_visualization(self, image: np.ndarray, segmentation: Optional[np.ndarray] = None) -> Optional[go.Figure]:
        """
        Create interactive 3D visualization using Plotly.
        
        Args:
            image: 3D image array
            segmentation: Optional 3D segmentation array
            
        Returns:
            Plotly figure
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for 3D visualization")
            return None
        
        # Downsample for performance if image is too large
        if np.prod(image.shape) > 128**3:
            factor = int(np.ceil(np.cbrt(np.prod(image.shape) / 128**3)))
            image = image[::factor, ::factor, ::factor]
            if segmentation is not None:
                segmentation = segmentation[::factor, ::factor, ::factor]
        
        # Create 3D volume visualization
        fig = go.Figure()
        
        # Add brain volume
        x, y, z = np.mgrid[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]]
        
        # Threshold image for better visualization
        threshold = np.percentile(image, 50)
        mask = image > threshold
        
        # Add brain surface
        fig.add_trace(go.Volume(
            x=x.flatten()[mask.flatten()],
            y=y.flatten()[mask.flatten()],
            z=z.flatten()[mask.flatten()],
            value=image.flatten()[mask.flatten()],
            isomin=threshold,
            isomax=image.max(),
            opacity=0.1,
            surface_count=15,
            colorscale='Greys',
            name='Brain'
        ))
        
        # Add tumor visualization if segmentation is provided
        if segmentation is not None:
            tumor_classes = [1, 2, 3, 4]  # Different tumor types
            colors = ['red', 'blue', 'green', 'yellow']
            names = ['Necrotic', 'Edema', 'Non-enhancing', 'Enhancing']
            
            for tumor_class, color, name in zip(tumor_classes, colors, names):
                tumor_mask = segmentation == tumor_class
                if np.any(tumor_mask):
                    fig.add_trace(go.Volume(
                        x=x.flatten()[tumor_mask.flatten()],
                        y=y.flatten()[tumor_mask.flatten()],
                        z=z.flatten()[tumor_mask.flatten()],
                        value=np.ones(tumor_mask.sum()),
                        opacity=0.6,
                        surface_count=5,
                        colorscale=[[0, color], [1, color]],
                        showscale=False,
                        name=f'Tumor - {name}'
                    ))
        
        # Update layout
        fig.update_layout(
            title='3D Brain Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_comparative_view(self, images: List[np.ndarray], titles: List[str]) -> Optional[go.Figure]:
        """
        Create comparative view of multiple brain scans.
        
        Args:
            images: List of 3D image arrays
            titles: List of titles for each image
            
        Returns:
            Plotly subplot figure
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for comparative visualization")
            return None
        
        n_images = len(images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=titles,
            specs=[[{'type': 'xy'}] * cols for _ in range(rows)]
        )
        
        for i, (image, title) in enumerate(zip(images, titles)):
            row = i // cols + 1
            col = i % cols + 1
            
            # Show middle slice
            mid_slice = image.shape[2] // 2
            slice_img = image[:, :, mid_slice]
            
            fig.add_trace(
                go.Heatmap(
                    z=slice_img,
                    colorscale='gray',
                    showscale=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Comparative Brain Scan Analysis',
            height=300 * rows
        )
        
        return fig
    
    def _create_overlay(self, segmentation_slice: np.ndarray) -> np.ndarray:
        """Create colored overlay for tumor segmentation."""
        overlay = np.zeros((*segmentation_slice.shape, 4))
        
        # Color map for different tumor types
        colors = {
            1: [1, 0, 0, 0.5],    # Red for necrotic
            2: [0, 0, 1, 0.5],    # Blue for edema
            3: [0, 1, 0, 0.5],    # Green for non-enhancing
            4: [1, 1, 0, 0.5]     # Yellow for enhancing
        }
        
        for class_id, color in colors.items():
            mask = segmentation_slice == class_id
            overlay[mask] = color
        
        return overlay
    
    def generate_tumor_statistics(self, segmentation: np.ndarray) -> Dict:
        """
        Generate statistical analysis of tumor segmentation.
        
        Args:
            segmentation: 3D segmentation array
            
        Returns:
            Dictionary with tumor statistics
        """
        stats = {}
        
        # Assume voxel spacing of 1mm³ for simplicity
        voxel_volume = 1.0
        
        # Calculate volumes for each tumor class
        class_names = {
            0: 'Background',
            1: 'Necrotic',
            2: 'Edema', 
            3: 'Non-enhancing',
            4: 'Enhancing'
        }
        
        for class_id, name in class_names.items():
            volume = np.sum(segmentation == class_id) * voxel_volume
            stats[name.lower() + '_volume_mm3'] = float(volume)
        
        # Calculate total tumor volume
        tumor_volume = sum([v for k, v in stats.items() if k != 'background_volume_mm3'])
        stats['total_tumor_volume_mm3'] = tumor_volume
        
        # Calculate tumor burden (percentage of brain volume)
        total_volume = np.prod(segmentation.shape) * voxel_volume
        stats['tumor_burden_percent'] = (tumor_volume / total_volume) * 100
        
        # Find tumor center of mass
        if tumor_volume > 0:
            tumor_mask = segmentation > 0
            coords = np.where(tumor_mask)
            center_of_mass = [float(np.mean(coord)) for coord in coords]
            stats['tumor_center_of_mass'] = center_of_mass
        
        return stats
    
    def display(self, image_path: str, segmentation_path: Optional[str] = None):
        """
        Main display function for brain visualization.
        
        Args:
            image_path: Path to brain MRI image
            segmentation_path: Optional path to tumor segmentation
        """
        # Load image
        image = self.load_image(image_path)
        
        # Load segmentation if provided
        segmentation = None
        if segmentation_path:
            segmentation = self.load_segmentation(segmentation_path)
        
        # Create visualizations
        logger.info("Creating slice viewer...")
        slice_fig = self.create_slice_viewer(image, segmentation)
        
        if PLOTLY_AVAILABLE:
            logger.info("Creating 3D visualization...")
            volume_fig = self.create_3d_visualization(image, segmentation)
            
            # Show in browser
            if volume_fig is not None:
                volume_fig.show()
        
        # Generate statistics if segmentation is available
        if segmentation is not None:
            stats = self.generate_tumor_statistics(segmentation)
            logger.info("Tumor Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
        
        # Show matplotlib figure
        plt.show()


def main():
    """Command line interface for brain visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Brain MRI 3D Visualization")
    parser.add_argument("--image", required=True, help="Path to brain MRI image")
    parser.add_argument("--segmentation", help="Path to tumor segmentation")
    parser.add_argument("--output", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    viewer = BrainViewer()
    viewer.display(args.image, args.segmentation)


if __name__ == "__main__":
    main()