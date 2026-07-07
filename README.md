# Deepfake Video Detection using ResNet50

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-%23000.svg?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-9.0+-blue.svg?logo=python&logoColor=white)


A web-based Deepfake Video Detection system that detects whether a video is Real or Fake using a ResNet50 deep learning model. The application extracts frames from an uploaded video, preprocesses them, performs frame-level prediction, and generates the final video classification.

# Project Overview

Deepfake technology uses Artificial Intelligence to generate realistic fake videos by manipulating facial expressions and identities. This project aims to detect manipulated videos using a fine-tuned ResNet50 convolutional neural network.

The system provides an easy-to-use web interface where users can upload a video and receive an instant prediction.

# 🚀 Features
- Upload video files through a web interface
- Automatic frame extraction
- Frame preprocessing
- Deepfake detection using ResNet50
- Final video-level prediction
- Simple Flask-based interface
- Confidence score visualization
  
# 🛠️ Technologies Used
- Python
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy
- ResNet50
- HTML
- CSS

# 📂 Project Structure
```
Deepfake-Detection/
│
├── static/
│   ├── style.css
│   ├── uploads/
│
├── templates/
│   └── index.html
|   └── result.html
│
├── app.py
├── df_notebook.ipynb
├── output.png 
└── README.md
```

# ⚙️ How It Works
- User uploads a video.
- Frames are extracted using OpenCV.
- Each frame is resized and normalized.
- The ResNet50 model predicts whether each frame is Real or Fake.
- Frame predictions are aggregated.
- The final prediction is displayed to the user.

# 📷 System Workflow
```
Upload Video
      │
      ▼
Extract Frames
      │
      ▼
Preprocess Frames
      │
      ▼
ResNet50 Prediction
      │
      ▼
Aggregate Frame Results
      │
      ▼
Display Final Prediction
```
