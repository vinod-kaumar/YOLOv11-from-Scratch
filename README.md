# PolypVision AI: Real-Time Polyp Detection

An end-to-end medical imaging project utilizing the state-of-the-art YOLOv11 architecture for the automated, real-time detection of polyps in endoscopic frames. This repository includes the complete custom training pipeline, evaluation metrics, and a FastAPI-powered web application designed for clinical deployment.

## 🚀 Project Overview

The goal of this project is to provide a high-precision, low-latency diagnostic aid for gastroenterologists. By training on a combination of open-source data and professional clinical datasets, the model achieves robust generalization across different medical equipment environments.

- **Model:** YOLOv11n (Nano)
- **Parameters:** ~2.3 Million
- **Inference Speed:** ~3.92 ms per image (254.8 FPS) on an RTX 4080 GPU
- **Accuracy:** 95.34% mAP@0.5 on Real Clinical Test Set

## 📂 Directory Structure

- `dataset.py`: Custom dataset loading, processing, and augmentation logic.
- `utils.py`: YOLOv11 architecture definitions, building blocks (C3k2, SPPF, PSA), and utility functions.
- `loss.py`: Custom composite loss function integrating BCE, Complete IoU (CIoU), and Distribution Focal Loss (DFL).
- `train.ipynb`: Primary notebook containing the model training loop, optimizer setup (AdamW), and hyperparameter scheduling.
- `evaluation.ipynb` & `evaluation_1.ipynb`: In-depth evaluation notebooks analyzing model performance on the complete test set and the specialized clinical test set.
- `inference.py`: High-speed inference script for processing individual frames and videos.
- `visualize.py`: Scripts for generating bounding box overlays and evaluation charts (e.g., Confusion Matrix, PR curves).
- `frontend/`: Full-stack web application. Includes a FastAPI backend (`app.py`) and a premium, responsive frontend (HTML/JS/CSS) featuring drag-and-drop uploads, live inference, and report generation.
- `Dockerfile` & `docker-compose.yml`: Containerization configurations for seamless, isolated deployment.
- `report.txt` & `ppt.txt`: Comprehensive documentation outlining project methodology, training details, and presentation content.

## 📊 Dataset

The training data was curated in two phases to ensure diversity:
1. **Phase 1 (Roboflow):** Foundational images with various shapes, sizes, and lighting conditions.
2. **Phase 2 (Clinical/Sir's Dataset):** High-quality, real-world clinical data to prevent overfitting and improve generalization.

**Total Training Images:** 6,502 | **Validation:** 902 | **Test:** 189

## 🧠 Architecture Highlights

- **Backbone:** YOLOv11 backbone optimized with Spatial Pyramid Pooling - Fast (SPPF).
- **Neck:** Incorporates Position-wise Spatial Attention (PSA) for superior feature fusion.
- **Head:** 3-scale Detect Head (8x, 16x, 32x strides) to accurately capture polyps of varying sizes.
- **Optimization:** Task Aligned Assigner (TAL) ensures high-quality positive anchor matching during training.

## 📈 Key Metrics (Clinical Test Set)

- **mAP@0.5:** 0.9534
- **Recall:** 0.9672
- **Precision:** 0.9465
- **F1-Score:** 0.9568

*Note: The model is tuned slightly toward sensitivity to ensure minimal false negatives, a critical requirement for medical screening applications.*

## 💻 Web Application

The repository includes a production-ready web interface designed with a clinical "Oasis" theme.
Features include:
- Real-time bounding box visualization.
- Adjustable confidence thresholds.
- Explainability tools (Grad-CAM visualization).
- Downloadable medical reports.

### Running the Application

**Using Docker:**
```bash
docker-compose up --build
```

**Using Python directly:**
```bash
pip install -r requirements.txt # (Assuming dependencies are listed)
python -m uvicorn frontend.app:app --host 0.0.0.0 --port 8000 --reload
```

## 📜 License
Please refer to the `LICENSE` file for details.
