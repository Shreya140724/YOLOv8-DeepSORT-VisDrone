# 🚁 YOLOv8-DeepSORT-VisDrone

<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python">

<img src="https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch">

<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-green">

<img src="https://img.shields.io/badge/OpenCV-4.x-blue?logo=opencv">

<img src="https://img.shields.io/badge/DeepSORT-Multi--Object--Tracking-orange">

<img src="https://img.shields.io/badge/Dataset-VisDrone2019-success">

<img src="https://img.shields.io/badge/License-MIT-yellow">

<img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?logo=googlecolab">

<img src="https://img.shields.io/badge/Status-Completed-brightgreen">

</p>

An end-to-end Computer Vision project for **real-time aerial object detection and multi-object tracking** using **YOLOv8** and **DeepSORT** on the **VisDrone2019** dataset.

---

## 📌 Overview

This project presents a complete pipeline for detecting and tracking multiple objects in drone videos. A custom **YOLOv8** detector is trained on the **VisDrone2019** dataset, while **DeepSORT** is used for assigning unique IDs and tracking objects across consecutive frames.

The project also includes image prediction, video prediction, tracker evaluation, visualization, and object statistics generation.

---

## ✨ Features

- 🚀 Custom YOLOv8 object detector
- 🎯 Multi-object tracking using DeepSORT
- 📷 Image inference
- 🎥 Video inference
- 🆔 Persistent object IDs
- 📊 Object counting
- 📈 Tracker evaluation
- 📉 Visualization dashboard
- 📁 CSV statistics generation
- ⚡ GPU accelerated inference

---

# Dataset

**Dataset:** VisDrone2019-DET

- Training Images: **6,471**
- Validation Images: **548**
- Classes: **10**

### Object Classes

- Pedestrian
- People
- Bicycle
- Car
- Van
- Truck
- Tricycle
- Awning-Tricycle
- Bus
- Motor

---

# Model Architecture

```

VisDrone Dataset
        │
        ▼
Dataset Preparation
        │
        ▼
YOLO Annotation Conversion
        │
        ▼
YOLOv8 Training
        │
        ▼
Image Prediction
        │
        ▼
Video Prediction
        │
        ▼
DeepSORT Tracking
        │
        ▼
Object Statistics
        │
        ▼
Evaluation
        │
        ▼
Visualization
```

---

# Folder Structure

```text
VisDrone_Project/
│
├── assets/
│   ├── detection.png
│   ├── tracking.png
│   ├── object_count_bar.png
│   ├── object_distribution_pie.png
│   └── top5_objects.png
│
├── outputs/
│   ├── tracking/
│   ├── statistics/
│   └── visualizations/
│
├── src/
│   ├── evaluate_detector.py
│   ├── evaluate_tracker.py
│   ├── predict_image.py
│   ├── predict_video.py
│   ├── tracker.py
│   └── visualize.py
│
├── runs/
│   └── YOLOv8_VisDrone/
│       └── results.png          
│
├── visdrone.yaml
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md```

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Deep Learning | PyTorch, YOLOv8 |
| Computer Vision | OpenCV, DeepSORT |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Environment | Google Colab |
| Dataset | VisDrone2019 |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/YOLOv8-DeepSORT-VisDrone.git

cd YOLOv8-DeepSORT-VisDrone
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Train YOLOv8 on the VisDrone dataset.

```python
python train.py
```

---

# Image Prediction

```python
python src/predict_image.py
```

---

# Video Prediction

```python
python src/predict_video.py
```

---

# Multi-Object Tracking

```python
python src/tracker.py
```

---

# Detector Evaluation

```python
python src/evaluate_detector.py
```

---

# Tracker Evaluation

```python
python src/evaluate_tracker.py
```

---

# Visualization

```python
python src/visualize.py
```

---

# Performance

| Metric | Value |
|---------|-------|
| Precision | **50.7%** |
| Recall | **38.4%** |
| mAP@50 | **37.7%** |
| mAP@50-95 | **22.0%** |

---

# Output

The project generates:

- ✅ Tracked Video
- ✅ Object Statistics CSV
- ✅ Detector Evaluation
- ✅ Tracker Evaluation
- ✅ Bar Chart
- ✅ Pie Chart
- ✅ Performance Reports

---

# Sample Outputs

## Object Detection

> Add detection screenshot here.

```
outputs/images/
```

---

## Multi-Object Tracking

> Add tracking video screenshot here.

```
outputs/tracking/
```

---

## Visualization

> Add bar chart and pie chart here.

```
outputs/visualizations/
```

---

## 🚀 Future Improvements

- Improve small object detection performance
- Experiment with YOLOv11 and RT-DETR
- Integrate ByteTrack for comparison
- Export the model to ONNX and TensorRT
- Deploy using Streamlit or FastAPI
- Support real-time webcam and drone stream inference

---

# Project Highlights

- End-to-end Computer Vision pipeline
- Custom YOLOv8 training
- Multi-object tracking
- Drone surveillance
- Object analytics
- Automated evaluation
- Visualization dashboard
- GPU accelerated inference

---

# Applications

- Smart City Monitoring
- Traffic Surveillance
- Drone Surveillance
- Public Safety
- Autonomous UAV Systems
- Intelligent Transportation
- Crowd Monitoring

---

# Author

**Shreya Sidabache**

M.Tech Artificial Intelligence

Computer Vision | Deep Learning | Machine Learning | Generative AI

---

# License

This project is licensed under the MIT License.

---

# Acknowledgements

- Ultralytics YOLOv8
- DeepSORT
- VisDrone2019 Dataset
- OpenCV
- PyTorch
