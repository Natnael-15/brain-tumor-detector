# Brain MRI Tumor Detector - Installation & Setup Guide

##  Quick Start

This guide will help you set up and run the Brain MRI Tumor Detector on your system.

##  System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 5GB free space
- **GPU**: CUDA-compatible GPU (optional, for training)

## 🛠️ Installation Steps

### 1. Environment Setup

First, ensure you have Python 3.8+ installed:
```bash
python --version
```

### 2. Install Dependencies

The project includes both required and optional dependencies:

**Required packages** (core functionality):
```bash
pip install numpy matplotlib scikit-learn opencv-python
```

**Optional packages** (full functionality):
```bash
pip install tensorflow torch torchvision nibabel plotly streamlit
```

**Complete installation** (all packages):
```bash
pip install -r requirements.txt
```

### 3. Verify Installation

Run the installation test:
```bash
python test_installation.py
```

Expected output:
```
 Brain MRI Tumor Detector - System Test
==================================================
 Basic Imports PASSED
 Project Structure PASSED  
 Configuration Files PASSED
 Module Imports PASSED
 Basic Functionality PASSED
 Sample Data Creation PASSED
 All tests passed!
```

##  Getting Started

### Option 1: Web Interface (Easiest)

1. Start the web application:
```bash
streamlit run app.py
```

2. Open your browser and go to: `http://localhost:8501`

3. Upload an MRI scan and analyze!

### Option 2: Command Line

1. Create sample data:
```bash
python scripts/download_data.py --dataset sample
```

2. Run preprocessing:
```bash
python src/main.py --mode preprocess --input data/raw/samples --output data/processed
```

3. Run analysis:
```bash
python src/main.py --mode predict --input data/processed/sample_brain_1.npy
```

### Option 3: Jupyter Notebooks

1. Install Jupyter:
```bash
pip install jupyter
```

2. Start Jupyter:
```bash
jupyter notebook
```

3. Open `notebooks/getting_started.md` for an interactive tutorial

##  Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'nibabel'`
**Solution**: Install medical imaging packages:
```bash
pip install nibabel SimpleITK pydicom
```

**Issue**: `ImportError: No module named 'torch'`
**Solution**: Install PyTorch:
```bash
pip install torch torchvision
```

**Issue**: `streamlit: command not found`
**Solution**: Install Streamlit:
```bash
pip install streamlit
```

**Issue**: CUDA out of memory
**Solution**: 
- Use CPU mode: Add `--device cpu` to commands
- Reduce batch size in config files
- Use smaller image sizes

### Memory Requirements

- **Minimum**: 4GB RAM (basic functionality)
- **Recommended**: 8GB RAM (full functionality)  
- **Optimal**: 16GB+ RAM (training and large datasets)

### GPU Support

For CUDA support (optional):
```bash
# NVIDIA GPU with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

##  Project Structure Overview

```
brain-tumor-detector/
├── src/                    # Source code
│   ├── data/              # Data preprocessing
│   ├── models/            # AI model definitions  
│   ├── training/          # Training scripts
│   ├── inference/         # Prediction engine
│   ├── visualization/     # 3D visualization
│   └── reports/           # Report generation
├── data/                  # Data storage
│   ├── raw/              # Original data
│   ├── processed/        # Preprocessed data
│   └── models/           # Trained models
├── config/               # Configuration files
├── notebooks/            # Jupyter notebooks
├── scripts/              # Utility scripts
├── tests/                # Test files
├── app.py               # Streamlit web app
├── requirements.txt     # Dependencies
└── README.md           # Documentation
```

##  Advanced Setup

### Custom Model Training

1. Prepare your dataset:
```bash
python scripts/download_data.py --dataset brats
```

2. Configure training:
Edit `config/training.yaml` with your settings

3. Start training:
```bash
python src/main.py --mode train --config config/training.yaml
```

### Development Setup

For contributing to the project:

1. Install development dependencies:
```bash
pip install -e .[dev]
```

2. Run tests:
```bash
pytest tests/
```

3. Code formatting:
```bash
black src/
flake8 src/
```

##  Next Steps

Once installed, you can:

1. **Try the Demo**: Use the web interface with sample data
2. **Explore Notebooks**: Check out interactive tutorials
3. **Train Models**: Use your own MRI datasets
4. **Customize**: Modify models and configurations
5. **Deploy**: Set up for production use

## 📞 Support

If you encounter issues:

1. Check this guide for solutions
2. Run `python test_installation.py` for diagnostics
3. Review the main README.md for detailed documentation
4. Create an issue on GitHub with error details

Happy analyzing! ✨