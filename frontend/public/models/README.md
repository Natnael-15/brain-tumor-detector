#  3D Brain Models Directory

This directory is where you should place your downloaded 3D brain models for use in the Brain MRI Tumor Detector.

## 📥 **Where to Download High-Quality Brain Models**

### **Recommended Sources:**

1. **Sketchfab.com**  **(BEST OPTION)**
   - URL: https://sketchfab.com/search?features=downloadable&q=human+brain+anatomy&type=models
   - Search for: "human brain anatomy", "medical brain", "anatomical brain"
   - Look for models with **"Download"** button
   - Choose **GLB** or **GLTF** format (works best with our viewer)
   - Many free models available from medical visualization artists

2. **CGTrader.com**
   - URL: https://www.cgtrader.com/free-3d-models/medical/anatomy/brain
   - Search for: "Brain Human with Anatomical Organs", "Brain and Nervous System"
   - Both free and premium options available
   - Supports OBJ, FBX, GLB formats

3. **Free3D.com**
   - URL: https://free3d.com/3d-models/brain
   - Look in Medical/Anatomy section
   - Usually has free brain models in multiple formats

4. **TurboSquid.com**
   - URL: https://www.turbosquid.com/Search/3D-Models/free/brain
   - Filter by "Free" models
   - High-quality medical models available

5. **Clara.io** 
   - URL: https://clara.io/library
   - Online 3D editor with model library
   - Can export in OBJ, FBX, GLB formats

## 📂 **Supported File Formats**

Our Advanced 3D Brain Viewer supports these formats:
-  **GLB** (Recommended - best performance)
-  **GLTF** (Recommended - with textures)
-  **OBJ** (With optional MTL material file)
-  **FBX** (With textures and animations)

##  **How to Add Models**

1. **Download** a brain model from one of the recommended sources
2. **Place** the file(s) in this `/models/` directory
3. **Name** your model files using one of these naming conventions:
   - `brain.glb` or `brain.gltf` (Primary model)
   - `brain_model.glb` or `brain_model.gltf`
   - `human_brain.glb` or `human_brain.gltf`
   - `anatomical_brain.glb` or `anatomical_brain.gltf`

##  **Example File Structure**

```
frontend/public/models/
├── brain.glb                    ← Main brain model (GLB format)
├── brain.gltf                   ← Alternative GLTF format
├── brain.obj                    ← OBJ geometry
├── brain.mtl                    ← OBJ materials (optional)
├── textures/                    ← Texture files (for GLTF/OBJ)
│   ├── brain_diffuse.jpg
│   ├── brain_normal.jpg
│   └── brain_specular.jpg
└── README.md                    ← This file
```

##  **What Makes a Good Brain Model**

For the best medical visualization experience, look for models with:

### **Anatomical Accuracy**
-  Realistic cerebral cortex with folding patterns (sulci and gyri)
-  Proper proportions and brain structure
-  Detailed surface geometry
-  Separate brain regions (frontal, parietal, temporal, occipital lobes)

### **Technical Quality**
-  High polygon count (>10,000 triangles for detail)
-  Clean geometry without holes or artifacts
-  Proper UV mapping for textures
-  Realistic materials and textures

### **Medical Features**
-  Anatomically correct size and proportions
-  Realistic brain color (pinkish-gray: #ffb3d1)
-  Surface details showing brain convolutions
-  Compatible with tumor overlay visualization

## 🔄 **Auto-Detection**

The Advanced 3D Brain Viewer will automatically:
1. **Scan** this directory for compatible models
2. **Load** the first available model automatically
3. **Display** all available models in the control panel
4. **Fallback** to a procedural brain if no models are found

## 🛠️ **Troubleshooting**

### Model Not Loading?
-  Check file format is supported (GLB, GLTF, OBJ, FBX)
-  Ensure file name matches expected patterns
-  Verify file isn't corrupted (try opening in 3D software)
-  Check browser console for loading errors

### Model Too Large/Small?
- The viewer automatically scales models to fit the viewport
- If scaling issues persist, try a different model

### Missing Textures?
- For GLTF/OBJ models, ensure texture files are in the same directory
- GLB files embed textures automatically (recommended)

##  **Medical Visualization Best Practices**

### For Clinical Use
- Use anatomically accurate models from medical sources
- Ensure proper lighting and material settings
- Verify tumor visualization accuracy
- Test with various lighting conditions

### For Education
- Choose detailed models with clear anatomical features
- Consider models with labeled regions
- Use models optimized for interactive viewing

## 📞 **Need Help?**

If you need assistance finding or implementing brain models:
1. Check the browser console for error messages
2. Verify the model file format and naming
3. Try different models from various sources
4. Contact the development team for technical support

---

**Current Status**: Ready for model import
**Last Updated**: October 8, 2025
**Compatible Formats**: GLB, GLTF, OBJ, FBX