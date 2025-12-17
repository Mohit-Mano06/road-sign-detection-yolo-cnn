import streamlit as st
# Set page config first - must be the first Streamlit command
st.set_page_config(page_title="Traffic Sign Detection", layout="wide")

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import io
import os
import pygame
from PIL import Image
import tempfile
import time
from datetime import datetime, timedelta
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load models and configurations
yolo_model = YOLO(r'C:\Users\delta\OneDrive\Desktop\Github\road-sign-detection-yolo-cnn\yolo\weights\train\best.pt') ## download best.pt from weights\train folder and copy path
cnn_model_instance = load_model(r'C:\Users\delta\OneDrive\Desktop\Github\road-sign-detection-yolo-cnn\cnn\model\cnn_model_custom_2.keras') ## download cnn_model_custom_2.keras and copy path

# Load class names
def load_yolo_class_names(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']

def load_cnn_class_names(csv_file):
    df = pd.read_csv(csv_file)
    return dict(zip(df['ClassId'], df['Name']))

yolo_class_names = load_yolo_class_names(r'C:\Users\delta\OneDrive\Desktop\Github\road-sign-detection-yolo-cnn\yolo\data\Road-Sign-detection-10\data.yaml') ## download data.yaml file from yolo\data and copy path
cnn_class_names = load_cnn_class_names(r'C:\Users\delta\OneDrive\Desktop\Github\road-sign-detection-yolo-cnn\cnn\data\traffic_sign_custom.csv') ## download trafficsigncustom csv from cnn\data and copy path

# Initialize pygame
pygame.mixer.init()

# Add these to your configuration
CONFIDENCE_THRESHOLD = 0.95
MIN_SIGN_SIZE = 32  # Minimum size in pixels
MAX_SIGN_SIZE = 400  # Maximum size in pixels

# Define ensemble models
def create_ensemble_models():
    # Create a base model for stacking
    base_models = [
        ('cnn', cnn_model_instance),  # Your CNN model
        ('yolo', yolo_model)           # Your YOLO model
    ]

    # Create a stacking model
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression()
    )

    # Create a voting model
    voting_model = VotingClassifier(
        estimators=base_models,
        voting='soft'  # Use soft voting for better performance
    )

    return stacking_model, voting_model

# Initialize ensemble models
stacking_model, voting_model = create_ensemble_models()

# Detection function (same as before)
def detect_signs(model, image):
    results = model(image)
    boxes = results[0].boxes.xyxy
    confidences = results[0].boxes.conf
    class_ids = results[0].boxes.cls
    confidence_threshold = 0.5
    filtered_boxes = [(box, conf, int(class_id)) for box, conf, class_id in zip(boxes, confidences, class_ids) if conf > confidence_threshold]
    return filtered_boxes

def resize_with_padding(image, target_size):
    old_size = image.shape[:2]
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image_resized = cv2.resize(image, (new_size[1], new_size[0]))
    new_image = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    y_offset = (target_size - new_size[0]) // 2
    x_offset = (target_size - new_size[1]) // 2
    new_image[y_offset:y_offset + new_size[0], x_offset:x_offset + new_size[1]] = image_resized
    return new_image

def preprocess_image(image):
    # Resize the image to the input size expected by the CNN
    target_size = (32, 32)  # Adjust this based on your CNN input size
    image_resized = cv2.resize(image, target_size)  # Resize to 32x32
    image_normalized = image_resized / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_normalized, axis=0)  # Add batch dimension

# Function to get ensemble predictions
def get_ensemble_prediction(cropped_image_rgb):
    # Preprocess the image for CNN
    processed_image = preprocess_image(cropped_image_rgb)

    # Get predictions from individual models
    cnn_prediction = cnn_model_instance.predict(processed_image)
    cnn_class_index = np.argmax(cnn_prediction, axis=1)[0]
    cnn_confidence = np.max(cnn_prediction)

    yolo_detections = detect_signs(yolo_model, cropped_image_rgb)  # Use the modified detect_signs function

    # Decision logic
    if yolo_detections:
        yolo_box, yolo_confidence, yolo_class_id = yolo_detections[0]  # Get the first detection
        yolo_class_name = yolo_class_names[yolo_class_id]

        # Apply decision logic
        if cnn_confidence < 0.8 and yolo_confidence > 0.8:
            final_prediction = yolo_class_name  # Trust YOLO if CNN is uncertain
        elif cnn_confidence >= 0.8 and cnn_confidence > yolo_confidence:
            final_prediction = cnn_class_names[cnn_class_index]  # Trust CNN if confident
        else:
            final_prediction = yolo_class_name  # Default to YOLO if both are uncertain
    else:
        final_prediction = cnn_class_names[cnn_class_index]  # Fallback to CNN if no YOLO detections

    return final_prediction

def detect_specific_signs(cropped_image, predicted_class):
    """Additional rules for specific sign detection"""
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    
    # Red color ranges (for No Parking signs)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 + red_mask2
    
    # Calculate color ratios
    red_ratio = np.sum(red_mask) / (red_mask.shape[0] * red_mask.shape[1])
    
    # Shape detection for specific signs
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=0, maxRadius=0)
    
    # Decision rules
    if predicted_class in ["No Parking", "Overtake Prohibited"]:
        if red_ratio > 0.3 and circles is not None:
            return "No Parking", 0.9
        elif red_ratio < 0.2 and circles is not None:
            return "Overtake Prohibited", 0.9
    
    return predicted_class, None

def process_image(uploaded_file):
    # Initialize variables with default values
    cropped_image_rgb = None
    final_classification = "No signs detected"

    # Decode image using OpenCV
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    boxes = detect_signs(yolo_model, image)
    yolo_image = image.copy()

    if len(boxes) > 0:
        pygame.mixer.music.load(r"C:\Users\delta\OneDrive\Desktop\YOLO CNN\Minor_Project_Code\Implementation\Audio_file\Double_beep.wav")
        pygame.mixer.music.play()

        for box, conf, class_id in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Add padding to bounding box
            padding = 30
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)

            # Draw bounding box
            cv2.rectangle(yolo_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            yolo_class_name = yolo_class_names[class_id]

            # Crop and process for CNN
            cropped_image = image[y1:y2, x1:x2]
            if cropped_image.size == 0:
                continue

            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

            # Get ensemble prediction
            ensemble_prediction = get_ensemble_prediction(cropped_image_rgb)

            # Final decision logic based on ensemble prediction
            if ensemble_prediction:
                final_classification = ensemble_prediction
            else:
                final_classification = yolo_class_name

            # Display both predictions
            st.markdown(f"**Final Classification:** {final_classification}")

        # Display the processed image
        st.image(cv2.cvtColor(yolo_image, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)

    else:
        st.image(cv2.cvtColor(yolo_image, cv2.COLOR_BGR2RGB), caption="No signs detected", use_column_width=True)

    return yolo_image, cropped_image_rgb, final_classification

def process_video(uploaded_video):
    # Create a temporary file to store the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    
    # Open the video file
    cap = cv2.VideoCapture(tfile.name)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer for saving processed video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"processed_video_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Create interface elements
    stframe = st.empty()
    result_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    # Add video controls
    col1, col2, col3 = st.columns(3)
    with col1:
        pause_button = st.button("⏸️ Pause")
    with col2:
        resume_button = st.button("▶️ Resume")
    with col3:
        stop_button = st.button("⏹️ Stop")
    
    # Add speed control
    speed_factor = st.slider("Playback Speed", min_value=0.25, max_value=2.0, value=1.0, step=0.25)
    
    # Initialize state variables
    paused = False
    stopped = False
    frame_count = 0
    detections_log = []
    
    while cap.isOpened() and not stopped:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
            # Detect signs in the frame
            boxes = detect_signs(yolo_model, frame)
            
            # Initialize a flag to track if any sign is detected with high confidence
            high_confidence_signs = []
            
            # Draw the current frame on the Streamlit app
            display_frame = frame.copy()
            
            if len(boxes) > 0:
                # Play double beep for detection
                pygame.mixer.music.load(r"C:\Users\delta\OneDrive\Desktop\YOLO CNN\Minor_Project_Code\Implementation\Audio_file\Double_beep.wav")
                pygame.mixer.music.play()
                
                for box, conf, class_id in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    yolo_class_name = yolo_class_names[class_id]
                    
                    # Add text background and label
                    text = f"{yolo_class_name} ({conf:.2f})"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_frame, (x1, y1-30), (x1 + text_width, y1), (0, 255, 0), -1)
                    cv2.putText(display_frame, text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Log detection only if confidence is above 80%
                    if conf > 0.8:
                        # Crop the detected sign from the frame
                        cropped_sign = frame[y1:y2, x1:x2]
                        
                        # Preprocess the cropped image for CNN
                        processed_sign = preprocess_image(cropped_sign)
                        
                        # Get classification from CNN
                        cnn_prediction = cnn_model_instance.predict(processed_sign)
                        cnn_class_index = np.argmax(cnn_prediction, axis=1)[0]
                        cnn_class_name = cnn_class_names.get(cnn_class_index, "Unknown")
                        cnn_confidence = np.max(cnn_prediction)
                        
                        # Determine final classification based on confidence
                        if conf > 0.87:
                            final_classification = yolo_class_name
                        else:
                            final_classification = cnn_class_name
                        
                        # Log high confidence sign
                        high_confidence_signs.append({
                            'frame': frame_count,
                            'time': frame_count / fps,
                            'yolo_class': yolo_class_name,
                            'cnn_class': cnn_class_name,
                            'final_classification': final_classification,
                            'cnn_confidence': cnn_confidence,
                            'yolo_confidence': conf
                        })
            
            # Write the processed frame with bounding boxes
            out.write(display_frame)
            
            # Display the modified frame with bounding boxes
            stframe.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        
            # Display results for high confidence signs
            if high_confidence_signs:
                for sign in high_confidence_signs:
                    result_placeholder.markdown(f"""
                        <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px; text-align: center;'>
                            <h2 style='color: #0066cc;'>Detection Result</h2>
                            <h3>YOLO Prediction: {sign['yolo_class']}</h3>
                            <h3>CNN Prediction: {sign['cnn_class']}</h3>
                            <p style='color: #666666;'>YOLO Confidence: {sign['yolo_confidence']:.2f}</p>
                            <p style='color: #666666;'>CNN Confidence: {sign['cnn_confidence']:.2f}</p>
                            <div style='padding: 10px; background-color: #e6f7ff; border-radius: 5px;'>
                                <h4 style='color: #0056b3;'>Final Classification: {sign['final_classification']}</h4>
                            </div>
                            <p style='color: #666666;'>Time: {sign['time']:.2f}s</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Handle control buttons
        if pause_button:
            paused = True
        if resume_button:
            paused = False
        if stop_button:
            stopped = True
    
    # Clean up
    cap.release()
    out.release()
    
    # Display detection summary
    if detections_log:
        st.markdown("### Detection Summary")
        df = pd.DataFrame(detections_log)
        st.dataframe(df)
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            # Download processed video
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="Download Processed Video",
                    data=f,
                    file_name=f"processed_video_{timestamp}.mp4",
                    mime="video/mp4"
                )
        with col2:
            # Download detection log
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Detection Log",
                data=csv,
                file_name=f"detection_log_{timestamp}.csv",
                mime="text/csv"
            )

# Update the CSS section with modern styling
st.markdown("""
    <style>
    /* Main container styles */
    .main-container {
        max-width: 85%;
        margin: 0 auto;
        padding: 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f8f9fa;
    }
    
    /* Header styles */
    .header {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        max-width: 85%;
        margin-left: auto;
        margin-right: auto;
        position: relative;
        overflow: hidden;
    }
    
    .header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 25%, transparent 25%),
                    linear-gradient(-45deg, rgba(255,255,255,0.1) 25%, transparent 25%);
        background-size: 60px 60px;
        opacity: 0.1;
    }
    
    .header h1 {
        margin: 0;
        padding: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        letter-spacing: 1px;
        color: #ffffff;
    }
    
    .header p {
        color: #e0e0e0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    /* Upload section styles */
    .upload-section {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        max-width: 85%;
        margin-left: auto;
        margin-right: auto;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Result card styles */
    .result-card {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-top: 2rem;
        max-width: 85%;
        margin-left: auto;
        margin-right: auto;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Image container styles */
    .image-container {
        border-radius: 20px;
        overflow: hidden;
        margin: 1.5rem auto;
        max-width: 85%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Custom column width */
    .custom-column {
        max-width: 85% !important;
        margin: 0 auto !important;
    }
    
    /* Override Streamlit's default container width */
    .stApp > header {
        max-width: 85% !important;
        margin: 0 auto !important;
    }
    
    .block-container {
        max-width: 85% !important;
        padding-left: 5rem !important;
        padding-right: 5rem !important;
    }
    
    /* Button styles */
    .stButton>button {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        color: white;
        border-radius: 30px;
        padding: 0.75rem 2.5rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Radio button styles */
    .stRadio>div {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* File uploader styles */
    .stFileUploader>div {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 2px dashed #1a237e;
    }
    
    /* Footer styles */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        color: white;
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 85%;
        margin-left: auto;
        margin-right: auto;
    }
    
    .footer p {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Progress bar styles */
    .stProgress>div>div>div {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
    }
    
    /* Slider styles */
    .stSlider>div>div>div {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
    }
    
    /* Text styles */
    .stMarkdown {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Detection result styles */
    .detection-result {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .detection-result h2 {
        color: #1a237e;
        margin-bottom: 1rem;
    }
    
    .detection-result h3 {
        color: #0d47a1;
        margin: 0.5rem 0;
    }
    
    .detection-result p {
        color: #666;
        margin: 0.5rem 0;
    }
    
    .confidence-bar {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        height: 4px;
        border-radius: 2px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Update the main layout section
st.markdown("""
    <div class="header">
        <h1>Road Sign Detection System</h1>
        <p style="font-size: 1.2rem; opacity: 0.9; margin-top: 0.5rem;">Using YOLO & CNN Hybrid Approach</p>
    </div>
    """, unsafe_allow_html=True)

# File type selector with improved styling
st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
        <h3 style="color: #1a237e; margin-bottom: 1rem;">Select Input Type</h3>
    </div>
    """, unsafe_allow_html=True)

file_type = st.radio("", ["Image", "Video"], key="file_type_selector", horizontal=True)

# Single upload section with improved styling
if file_type == "Image":
    st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
            <h3 style="color: #1a237e; margin-bottom: 1rem;">Upload an Image</h3>
            <p style="color: #666; margin-bottom: 1rem;">Supported formats: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "", 
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )
    if uploaded_file is not None:
        yolo_image, cnn_image, final_classification = process_image(uploaded_file)
else:
    st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
            <h3 style="color: #1a237e; margin-bottom: 1rem;">Upload a Video</h3>
            <p style="color: #666; margin-bottom: 1rem;">Supported formats: MP4, AVI, MOV</p>
        </div>
        """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "", 
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )
    if uploaded_file is not None:
        process_video(uploaded_file)

# Footer with improved styling
st.markdown("""
    <div class="footer">
        <p>Developed by Mohit Manoharan</p>
        <p style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">© 2024 Traffic Sign Detection System</p>
    </div>
    """, unsafe_allow_html=True)

def is_turn_left_sign(cropped_image):
    # Convert the cropped image to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Use Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
    
    # Analyze the lines to determine if they correspond to a left turn sign
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Example logic: Check the slope of the line
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
            if slope < 0:  # Negative slope might indicate a left turn
                return True
    
    return False