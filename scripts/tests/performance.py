"""Module for trying out the performance of YOLOv8."""
from cv2 import imshow, waitKey
from ultralytics import YOLO

from analysis.util.yolov8 import plot, predict

WEIGHT_FILES = [
    "weights/yolov8n.pt",
    "weights/yolov8s.pt",
    "weights/yolov8m.pt",
    "weights/yolov8l.pt",
    "weights/yolov8x.pt",
]

results = None
for weight_file in WEIGHT_FILES:
    model = YOLO(weight_file)
    results = predict(model, "images/person.png")
if results is not None:
    imshow("Result", plot(results[0]))
    waitKey(0)
