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

## 📂 Folder Structure

road-sign-detection/
│
├── cnn/
│   ├── model.ipynb              # CNN training notebook
│   ├── predict.py               # CNN inference script
│   ├── labels.csv               # Class ID ↔ Sign name mapping
│   └── README.md                # CNN-specific details
│
├── yolo/
│   ├── yolomodel.ipynb           # YOLO training & inference notebook
│   ├── data/
│   │   ├── images/               # Training & validation images
│   │   └── labels/               # YOLO annotation files
│   ├── runs/                     # YOLO outputs (ignored in git)
│   └── README.md                 # YOLO-specific details
│
├── app/
│   ├── streamlit_app.py          # Streamlit web application
│   └── utils.py                  # Helper functions
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files/folders ignored by Git
├── README.md                     # Main project documentation
└── LICENSE


