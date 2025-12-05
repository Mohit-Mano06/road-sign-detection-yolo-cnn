# Road Sign Detection – YOLO + CNN Hybrid Model

This project implements a hybrid **YOLO + CNN based traffic sign detection system**.  
YOLO is used for **real-time object detection**, while a custom CNN model classifies the cropped sign into its appropriate category.

---

## 🚀 Features
- Real-time road sign detection using YOLOv5/YOLOv8  
- CNN classifier for fine-grained traffic sign recognition  
- End-to-end hybrid pipeline:  
  YOLO → Crop Sign → CNN → Predicted Label  
- Streamlit web app for demo  
- Trained on custom traffic-sign dataset

---

## 🧠 Project Architecture

Camera -> YOLO Detection -> Bounding Box -> CNN Classification -> Output Sign/Result (with Audiio Bizzer)

---

## 📂 Folder Structure

