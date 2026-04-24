# PolypVision AI: Real-Time Polyp Detection

An end-to-end medical imaging project utilizing the state-of-the-art YOLOv11 architecture for the automated, real-time detection of polyps in endoscopic frames. This repository includes the complete custom training pipeline, evaluation metrics, and a FastAPI-powered web application designed for clinical deployment.

## 🚀 Project Overview

The goal of this project is to provide a high-precision, low-latency diagnostic aid for gastroenterologists. By training on a combination of open-source data and professional clinical datasets, the model achieves robust generalization across different medical equipment environments.

- **Model:** YOLOv11n (Nano)
- **Parameters:** ~2.3 Million
- **Accuracy (mAP@0.5):** 93.43%
- **Inference Speed:** ~892.7 FPS (GPU) / ~10.48 FPS (CPU)

## 📂 Directory Structure

- `dataset.py`: Custom dataset loading, processing, and augmentation logic.
- `utils.py`: YOLOv11 architecture definitions, building blocks (C3k2, SPPF, PSA), and utility functions.
- `loss.py`: Custom composite loss function integrating BCE, Complete IoU (CIoU), and Distribution Focal Loss (DFL).
- `train.ipynb`: Primary notebook containing the model training loop, optimizer setup (AdamW), and hyperparameter scheduling.
- `evaluation.ipynb` & `evaluation_1.ipynb`: In-depth evaluation notebooks analyzing model performance on the complete test set and the specialized clinical test set.
- `inference.py`: High-speed inference script for processing individual frames and videos.
- `visualize.py`: Scripts for generating bounding box overlays and evaluation charts (e.g., Confusion Matrix, PR curves).
- `frontend/`: Full-stack web application. Includes a FastAPI backend (`app.py`) and a premium, responsive frontend (HTML/JS/CSS) featuring drag-and-drop uploads, live inference, and report generation.
- `report.txt` & `ppt.txt`: Comprehensive documentation outlining project methodology, training details, and presentation content.

## 📊 Dataset

The training data was curated in two phases to ensure diversity:
1. **Phase 1 (Roboflow):** Foundational images with various shapes, sizes, and lighting conditions.
2. **Phase 2 (Clinical/Sir's Dataset):** High-quality, real-world clinical data to prevent overfitting and improve generalization.

**Total Images:** 9,035 | **Training:** 6,502 | **Validation:** 902 | **Test:** 1,631

## 🧠 Architecture Highlights

- **Backbone:** YOLOv11 backbone optimized with Spatial Pyramid Pooling - Fast (SPPF).
- **Neck:** Incorporates Position-wise Spatial Attention (PSA) for superior feature fusion.
- **Head:** 3-scale Detect Head (8x, 16x, 32x strides) to accurately capture polyps of varying sizes.
- **Optimization:** Task Aligned Assigner (TAL) ensures high-quality positive anchor matching during training.

## 📈 Key Metrics

- **mAP@0.5:** 0.9343 (93.43%)
- **Precision:** 0.9749 (97.49%)
- **Recall:** 0.9299 (92.99%)
- **F1 Score:** 0.9519

### Speed:
- **CPU Inference:** ~10.48 FPS (Robust for edge deployment).
- **GPU Inference:** ~892.7 FPS (Ultra-fast screening).

## 💻 Web Application

The repository includes a production-ready web interface designed with a clinical "Oasis" theme.
Features include:
- Real-time bounding box visualization.
- Adjustable confidence thresholds.
- Explainability tools (Grad-CAM visualization).
- Downloadable medical reports.

### Running the Application

```bash
pip install -r requirements.txt # (Assuming dependencies are listed)
python -m uvicorn frontend.app:app --host 0.0.0.0 --port 8000 --reload
```

## 📜 License
Please refer to the `LICENSE` file for details.
