from ultralytics import YOLO

from util import run_detection

model = YOLO("weights/yolov8n-pose.pt")
run_detection(model)
