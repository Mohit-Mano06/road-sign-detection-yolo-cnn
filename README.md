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
- Added Docker for easy setup and execution without manual dependency installation

---

## 🧠 Project Architecture

Camera -> YOLO Detection -> Bounding Box -> CNN Classification -> Output Sign/Result (with Audio Buzzer)

---

## Prerequisites 

- Docker to be installed on your system

## ▶️ How to Run the Application


- Clone the repository or download ZIP
``` bash
  git clone https://github.com/your-username/road-sign-detection.git
```

- Navigate to the app folder 
```bash
  cd road-sign-detection/app
```

- Run Docker command to build the image
```bash
  docker build -t road-sign-streamlit .
  docker run -p 8501:8501 road-sign-streamlit
```

- Once container is up and running

- Open browser and visit [http://localhost:8501](http://localhost:8501)


## Notes

- The Docker container includes only the Streamlit inference application.

- Model files, configuration files, and assets required for inference are packaged inside the container.

- Training code and datasets are intentionally excluded to keep the container lightweight.

- Docker eliminates environment and dependency conflicts across systems.