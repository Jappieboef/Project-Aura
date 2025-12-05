<div align="center">

# 🌌 Project Aura  
### **Real-Time Emotion Detection From Video Using YOLO**

<img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge">
<img src="https://img.shields.io/badge/OpenCV-Enabled-green?style=for-the-badge">
<img src="https://img.shields.io/badge/YOLO-Emotion_Model-orange?style=for-the-badge">
<br>

<p style="font-size: 1.2rem; max-width: 700px; text-align: center;">
Project Aura processes video streams in real time, detects human faces, 
classifies emotional states, and overlays predictions directly onto video frames.
A flexible, modular pipeline built for research, experimentation, and development.
</p>

</div>

---

## 🌟 Features

-  **Face Detection** powered by a YOLOv8 face model  
-  **Emotion Classification** using a trained YOLO emotion model  
-  **Real-Time Annotation** with bounding boxes + emotion labels  
-  **Modular Architecture** with separate detection/classification modules  
-  **Optional Frame Extraction** for dataset creation and analysis  

---

##  Project Structure

```plaintext
Project Aura/
│
├── detect_faces.py           # Real-time face & emotion pipeline
├── classify_emotions.py      # Emotion classification module
├── main.py                   # Frame extraction utility
│
├── model.pt                  # Emotion model weights (optional)
├── yolov8n-face.pt           # Face detection model weights (optional)
│
├── face_crops/               # Auto-generated (ignored)
├── frames/                   # Auto-generated (ignored)
│
└── README.md
