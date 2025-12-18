# Road Sign Detection – YOLO + CNN Hybrid Model

This project implements a hybrid **YOLO + CNN based traffic sign detection system**.  
YOLO is used for **real-time object detection**, while CNN model classifies the road sign into its appropriate category.

---

## 🚀 Features
- Real-time road sign detection using YOLOv5/YOLOv8  
- CNN classifier for fine-grained traffic sign recognition  
- End-to-end hybrid pipeline:  
  YOLO → Crop Sign → CNN → Predicted Label  
- Streamlit web app for demo  
- Trained on custom traffic-sign dataset **from Roboflow**

---

## 🧠 Project Architecture

Camera -> YOLO Detection -> Bounding Box -> CNN Classification -> Output Sign/Result (with Audio Buzzer)

---

## How to run the Application 

- Clone the repository or download ZIP
- Navigate to the app folder 
- Install the dependencies using
```bash
  pip install -r requirements-app.txt
```
- In modern_interface.py file , replace the path of the CNN & YOLO [Line 28 & 29], Data.yaml & CNN csv file [Line 41 & 42] and Audio Beep wav (Line 177)
- Run Streamlit App 
```bash
  streamlit run modern_interface.py
```

## 📂 Folder Structure
