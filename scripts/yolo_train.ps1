# Commands to train YOLOv8 model on a windows system
$config_path = "datasets\on-shelf-stock-availability-ox04t\data.yaml"
yolo.exe task=detect mode=train model=yolov8n.pt data=$config_path epochs=150 workers=12