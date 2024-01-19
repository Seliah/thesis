# Wave your wand (or keyboard) to get started!
from ultralytics import YOLO

from analysis.util import run_detection

# Cast a spell to summon the model
model = YOLO("weights/best.pt")

# Tweak the magical parameters
model.overrides["conf"] = 0.25  # NMS confidence threshold
model.overrides["iou"] = 0.45  # NMS IoU threshold
model.overrides["agnostic_nms"] = False  # NMS class-agnostic
model.overrides["max_det"] = 1000  # maximum number of detections per image

run_detection(model)
