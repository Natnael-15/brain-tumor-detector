# 🏗️ Brain MRI Tumor Detector - Architecture Guide

## 📁 **Project Structure Clarification**

This project has evolved through multiple phases, resulting in the following architecture:

```
brain-tumor-detector/
├── 🆕 backend/                 # Modern FastAPI Backend (Phase 2/3)
│   ├── main.py                 # FastAPI server entry point
│   ├── services/               # Business logic services
│   └── uploads/                # File upload storage
│
├── 🆕 frontend/                # Modern Next.js Frontend (Phase 2/3)  
│   ├── src/                    # React components & logic
│   │   ├── app/                # Next.js 14 app directory
│   │   ├── components/         # React components
│   │   ├── hooks/              # Custom React hooks
│   │   └── lib/                # Utilities & services
│   ├── package.json            # Node.js dependencies
│   └── next.config.js          # Next.js configuration
│
├── 📦 legacy-backend/          # Original Phase 1 Implementation
│   ├── main.py                 # CLI entry point
│   ├── models/                 # AI model implementations
│   ├── data/                   # Data processing
│   ├── training/               # Model training scripts
│   ├── inference/              # Prediction engine
│   └── visualization/          # Legacy visualization
│
├── 📊 data/                    # Training & test data
├── 🔧 config/                  # Configuration files
├── 📚 docs/                    # Documentation
├── 🧪 tests/                   # Test suites
└── 📋 scripts/                 # Utility scripts
```

## 🚀 **Current Active Architecture**

### **Production System (Phases 2 & 3)**
- **Backend**: `backend/main.py` (FastAPI + WebSocket)
- **Frontend**: `frontend/src/` (Next.js 14 + React 18)
- **Services**: Real-time analysis, 3D visualization, clinical reporting

### **Legacy System (Phase 1)**  
- **Backend**: `legacy-backend/main.py` (CLI-based)
- **Purpose**: Research, training, batch processing
- **Models**: nnU-Net, Medical Vision Transformers, BraTS integration

## 🔄 **Why Two `src` Directories Were Confusing**

Previously, there were two `src/` directories:
1. **Root `src/`** → Now `legacy-backend/` (Phase 1 implementation)
2. **`frontend/src/`** → Still `frontend/src/` (Modern React frontend)

This caused confusion about which implementation was active.

## 🎯 **How to Use Each System**

### **🌐 Modern Web Interface (Recommended)**
```bash
# Terminal 1: Start backend
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start frontend  
cd frontend
npm run dev
```
**Access**: http://localhost:3000

### **🖥️ Legacy CLI Interface**
```bash
# Preprocessing
python legacy-backend/main.py --mode preprocess --input data/raw --output data/processed

# Training
python legacy-backend/main.py --mode train --input data/processed --output data/models

# Prediction
python legacy-backend/main.py --mode predict --input scan.nii --model data/models/best_model.pth
```

## 🏥 **For Medical Professionals**

**Use the Modern Web Interface** (localhost:3000):
- ✅ Real-time analysis with WebSocket updates
- ✅ Clinical-grade medical reporting with WHO standards
- ✅ Interactive 3D brain visualization with tumor overlay
- ✅ Professional medical dashboard
- ✅ DICOM and NIfTI support
- ✅ Analysis history and report generation

## 🔬 **For Researchers & Developers**

**Use the Legacy CLI** for:
- 🧠 Training new AI models (nnU-Net, Medical ViT)
- 📊 Batch processing large datasets  
- 🔬 Research and experimentation
- 📈 Performance benchmarking
- 🏗️ Custom model development

## 🔧 **Integration Points**

The modern backend (`backend/`) automatically imports models from `legacy-backend/` when available:
- ✅ Seamless integration between Phase 1 models and modern API
- ✅ Graceful fallback to mock models for development
- ✅ Real AI model predictions via FastAPI endpoints

## 📚 **Migration Path**

1. **Current**: Use modern web interface for clinical work
2. **Development**: Use legacy CLI for model training/research  
3. **Future**: Gradually migrate CLI functionality to web interface
4. **Production**: Deploy modern system with Docker Compose

---

This architecture provides the best of both worlds: a modern, clinical-grade web interface for daily use, and a powerful CLI system for research and development! 🎉