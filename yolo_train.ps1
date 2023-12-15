$p = "C:\Users\selia\Code\thesis\data.yaml"
yolo.exe task=detect mode=train model=yolov8n.pt data=$p epochs=3 imgsz=416