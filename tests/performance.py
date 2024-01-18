from cv2 import imshow, waitKey
from ultralytics import YOLO

WEIGHT_FILES = [
    "weights/yolov8n.pt",
    "weights/yolov8s.pt",
    "weights/yolov8m.pt",
    "weights/yolov8l.pt",
    "weights/yolov8x.pt",
]

for weight_file in WEIGHT_FILES:
    results = YOLO(weight_file).predict("images/person.png")
imshow("Result", results[0].plot())
waitKey(0)
