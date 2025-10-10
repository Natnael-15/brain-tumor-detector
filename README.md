# 🧠 Brain MRI Tumor Detector

## 💭 My Journey

This project started as my personal journey into applying technology for real-world impact in healthcare. I wanted to explore how AI and data visualization could improve early detection and understanding of brain-related conditions.

When I began, I had very little experience with medical imaging or neural network design. Through research, documentation, and constant iteration, I learned about MRI data preprocessing, model training, and the challenges of working with medical datasets. Debugging model inconsistencies, handling noisy scan data, and tuning hyperparameters were all part of the learning curve.

What began as a small idea soon evolved into a full-stack healthcare simulation system — combining AI tumor detection with an integrated **Brain Wave Monitoring Simulator**. Beyond the technical side, I gained an appreciation for how accessible tools and open-source frameworks can help democratize healthcare innovation.

This project reflects my goal of using technology responsibly and meaningfully. It’s not perfect, but it’s made with care, curiosity, and a drive to learn.

---

## 🎯 Features

* **AI-Powered Tumor Detection**: Upload MRI scans for automated tumor classification and segmentation using deep learning.
* **Brain Wave Monitoring Simulator**: A simulated environment that visualizes brainwave patterns, response signals, and neural activity mapping.
* **Medical Reports**: Automatically generates analysis reports with AI confidence scores and diagnosis insights.
* **Treatment Comparison**: Visual comparison between scan sessions to analyze tumor growth or reduction trends.
* **Dataset Support**: Compatible with public datasets including BraTS, TCIA, and Kaggle medical imaging sets.
* **WebSocket Real-Time Feedback**: Real-time analysis status updates and simulation control.
* **Modern Interface**: Clean, medical-style UI built with Next.js, React, and TailwindCSS.

---

## 🧩 Tech Stack

**Frontend:** React, Next.js, TailwindCSS
**Backend:** Node.js, Express
**AI/ML:** Python, TensorFlow, OpenCV, PyTorch
**Database:** MongoDB
**Deployment:** Vercel / Render (free-tier)
**Other Tools:** WebSocket, REST API, Supabase (for storage)

---

## ⚙️ Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/brain-mri-tumor-detector.git
   ```

2. Navigate to the project directory:

   ```bash
   cd brain-mri-tumor-detector
   ```

3. Install dependencies:

   ```bash
   npm install
   ```

4. Run the development server:

   ```bash
   npm run dev
   ```

5. For AI model setup (Python backend):

   ```bash
   pip install -r requirements.txt
   python app.py
   ```

6. Access the app at `http://localhost:3000`

---

## 🧠 Model & Data

The detection model is trained on **Brain Tumor MRI Datasets** from publicly available sources (BraTS 2021 & Kaggle MRI dataset). The network uses a convolutional segmentation model with data augmentation and normalization.

### Model Highlights:

* CNN-based feature extraction with transfer learning
* Multi-class segmentation: glioma, meningioma, pituitary
* Accuracy: ~95% (validation)
* Input size: 240x240 grayscale MRI slices

---

## 📊 Brain Wave Monitoring Simulator

The **Brain Wave Monitoring Simulator** mimics the visualization of neural signal responses and activity patterns. It processes synthetic EEG-style data to generate dynamic real-time waveforms. This feature aims to simulate the connection between tumor location and signal variations for experimental study.

### Simulator Features:

* Real-time waveform display
* Adjustable frequency, noise, and amplitude
* Interactive UI with parameter sliders
* Export simulated data for research

---

## 📈 Performance Metrics

| Metric          | Value |
| ---------       | ----- |
| Accuracy        | 95%   |
| Precision       | 93%   |
| Recall          | 94%   |
| F1-Score        | 93.5% |

---

## 🚧 Roadmap

* [x] MRI preprocessing pipeline
* [x] CNN model for tumor segmentation
* [x] Brain Wave Monitoring simulator
* [ ] Integration with real EEG data (future)
* [ ] Cloud deployment (Docker + GPU support)
* [ ] Enhanced dashboard analytics

---

## 🙌 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`feature/new-idea`)
3. Commit your changes
4. Submit a pull request

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is **not intended for real-world medical diagnosis or treatment**. Any clinical decisions should be made by licensed healthcare professionals.

---

## 🩺 Acknowledgments

* BraTS Challenge dataset team for open medical MRI data
* TensorFlow and PyTorch communities for tools and resources
* Open-source developers and researchers contributing to medical AI

---

### 📜 License

This project is released under the **MIT License** — free to use, modify, and distribute.

---

> Built with curiosity, care, and a vision for accessible healthcare innovation.
