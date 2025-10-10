# Brain MRI Tumor Detection - Getting Started

Welcome to the Brain MRI Tumor Detection project! This notebook will help you get started with analyzing brain MRI scans using our AI models.

##  Prerequisites

Before running this notebook, make sure you have:
- Python 3.8+ installed
- All required dependencies from requirements.txt
- Sample MRI data (can be generated using our scripts)

##  Quick Start

Let's start by importing the necessary libraries and loading a sample brain MRI scan.

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd().parent / 'src'))

# Import our modules
from data.preprocess import MRIPreprocessor
from inference.predict import TumorPredictor
from visualization.viewer import BrainViewer

print(" Imports successful!")
```

## 📂 Data Loading

First, let's load a sample MRI scan and visualize it.

```python
# Initialize the preprocessor
preprocessor = MRIPreprocessor()

# Load sample data (you'll need to run the download script first)
data_dir = Path.cwd().parent / 'data' / 'raw' / 'samples'

if data_dir.exists():
    sample_files = list(data_dir.glob('*.npy'))
    if sample_files:
        sample_file = sample_files[0]
        print(f"Loading: {sample_file}")
        
        # Load the MRI data
        mri_data = np.load(sample_file)
        print(f"MRI shape: {mri_data.shape}")
        print(f"Data range: {mri_data.min():.3f} to {mri_data.max():.3f}")
    else:
        print(" No sample data found. Run 'python scripts/download_data.py' first")
        # Create dummy data for demonstration
        mri_data = np.random.rand(128, 128, 128)
else:
    print(" Data directory not found. Creating dummy data for demonstration")
    mri_data = np.random.rand(128, 128, 128)
```

## 🖼️ Visualization

Let's visualize the MRI scan in different views.

```python
# Display middle slices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

mid_x, mid_y, mid_z = [s // 2 for s in mri_data.shape]

# Sagittal view
axes[0].imshow(mri_data[mid_x, :, :], cmap='gray')
axes[0].set_title('Sagittal View')
axes[0].axis('off')

# Coronal view  
axes[1].imshow(mri_data[:, mid_y, :], cmap='gray')
axes[1].set_title('Coronal View')
axes[1].axis('off')

# Axial view
axes[2].imshow(mri_data[:, :, mid_z], cmap='gray')
axes[2].set_title('Axial View')
axes[2].axis('off')

plt.tight_layout()
plt.show()

print(" Basic visualization complete!")
```

## 🔍 Preprocessing

Now let's preprocess the MRI data for analysis.

```python
# Normalize the data
normalized_data = preprocessor.normalize_intensity(mri_data, method='percentile')

# Resample to target shape if needed
if mri_data.shape != (128, 128, 128):
    processed_data = preprocessor.resample_image(normalized_data, (128, 128, 128))
else:
    processed_data = normalized_data

print(f" Preprocessing complete!")
print(f"Original shape: {mri_data.shape}")
print(f"Processed shape: {processed_data.shape}")
print(f"Processed range: {processed_data.min():.3f} to {processed_data.max():.3f}")
```

## Model Analysis

Let's run the tumor detection model on the processed data.

```python
# Note: This would normally use a trained model
# For demonstration, we'll simulate the analysis

print("Running tumor detection analysis...")

# Simulate tumor detection
has_tumor = np.random.choice([True, False], p=[0.3, 0.7])
confidence = np.random.uniform(0.75, 0.98)
tumor_type = np.random.choice(['Glioma', 'Meningioma', 'Pituitary', 'No Tumor'])

if has_tumor:
    print(f"TUMOR DETECTED")
    print(f"   Type: {tumor_type}")
    print(f"   Confidence: {confidence:.1%}")
    
    # Create a sample tumor mask for visualization
    tumor_mask = np.zeros_like(processed_data)
    center = tuple(s // 2 for s in processed_data.shape)
    
    # Add sample tumor region
    z, y, x = np.ogrid[:processed_data.shape[0], :processed_data.shape[1], :processed_data.shape[2]]
    mask = ((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2) <= 15**2
    tumor_mask[mask] = 1
    
    # Visualize with tumor overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show middle slices with tumor overlay
    axes[0].imshow(processed_data[mid_x, :, :], cmap='gray')
    axes[0].contour(tumor_mask[mid_x, :, :], colors='red', linewidths=2)
    axes[0].set_title('Sagittal View with Tumor')
    axes[0].axis('off')
    
    axes[1].imshow(processed_data[:, mid_y, :], cmap='gray')
    axes[1].contour(tumor_mask[:, mid_y, :], colors='red', linewidths=2)
    axes[1].set_title('Coronal View with Tumor')
    axes[1].axis('off')
    
    axes[2].imshow(processed_data[:, :, mid_z], cmap='gray')
    axes[2].contour(tumor_mask[:, :, mid_z], colors='red', linewidths=2)
    axes[2].set_title('Axial View with Tumor')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
else:
    print(f" NO TUMOR DETECTED")
    print(f"   Confidence: {confidence:.1%}")
```

## Analysis Results

Let's generate some analysis statistics.

```python
# Calculate some basic statistics
print("ANALYSIS RESULTS")
print("=" * 40)
print(f"Volume analyzed: {np.prod(processed_data.shape):,} voxels")
print(f"Brain volume estimate: {np.sum(processed_data > 0.1):,} mm³")

if has_tumor:
    tumor_volume = np.sum(tumor_mask) 
    print(f"Tumor volume: {tumor_volume:,} mm³")
    print(f"Tumor burden: {tumor_volume / np.sum(processed_data > 0.1) * 100:.1f}%")

print(f"Processing completed in simulated time")
```

## Report Generation

Finally, let's create a simple analysis report.

```python
from datetime import datetime

# Generate a simple report
report = f"""
BRAIN MRI ANALYSIS REPORT
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FINDINGS:
"""

if has_tumor:
    report += f"""
- ABNORMAL: Tumor detected
- Classification: {tumor_type}
- Confidence: {confidence:.1%}
- Estimated Volume: {tumor_volume:,} mm³

RECOMMENDATIONS:
- Consultation with neuro-oncologist recommended
- Additional imaging studies may be required
- Clinical correlation advised
"""
else:
    report += f"""
- NORMAL: No significant abnormalities detected  
- Confidence: {confidence:.1%}

RECOMMENDATIONS:
- Routine follow-up as clinically indicated
- No immediate intervention required
"""

report += """
DISCLAIMER:
This analysis is for research and educational purposes only.
All clinical decisions must be made by qualified medical professionals based on 
comprehensive evaluation of patient history, symptoms, and additional diagnostic tests.
"""

print(report)

# Save report to file
report_file = Path.cwd().parent / 'output' / 'analysis_report.txt'
report_file.parent.mkdir(parents=True, exist_ok=True)

with open(report_file, 'w') as f:
    f.write(report)

print(f"📄 Report saved to: {report_file}")
```

##  Next Steps

Congratulations! You've completed the basic brain MRI analysis workflow. Here are some next steps:

1. **Explore Advanced Features**: Try the 3D visualization tools
2. **Train Custom Models**: Use your own datasets for training
3. **Batch Processing**: Analyze multiple scans at once
4. **Web Interface**: Try the Streamlit app for a user-friendly interface

## 📚 Additional Resources

- **Documentation**: Check the `docs/` folder for detailed guides
- **Example Notebooks**: Explore other notebooks in this directory
- **Model Training**: See `notebooks/model_training.ipynb`
- **Data Analysis**: Check `notebooks/data_analysis.ipynb`

Happy analyzing! ✨