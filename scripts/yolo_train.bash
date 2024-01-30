# Commands to train YOLOv8 model on a Linux system
yolo task=detect \
  mode=train \
  model=yolov8s.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=416
