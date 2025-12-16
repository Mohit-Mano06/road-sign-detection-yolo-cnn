# YOLO Traffic Sign Detection

This module implements traffic sign detection using the YOLO (You Only Look Once) object detection framework.  
The model is trained to detect traffic signs in road images by predicting bounding boxes and class IDs.

- `yolomodel.ipynb` contains the training and detection code
- Images and labels are stored in the `data/` directory
- Dataset configuration is defined in `data.yaml`
- Training outputs and weights are generated in the `runs/` directory
- Weight from best.pt is found in weights folder

YOLO is used for localization, while classification is handled separately using a CNN.

