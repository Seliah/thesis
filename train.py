import rich
rich.print("Booting up! 🚀")
rich.print("Loading CV libs...")
import torch
from ultralytics import YOLO, checks


checks()

# # Check for CUDA device and set it
# # See https://github.com/ultralytics/ultralytics/issues/3084#issuecomment-1806617276
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Using device: {device}')
# torch.cuda.set_device(0)

# Load the model
model = YOLO('yolov8n.pt')

# Train the model
model.train(data='datasets\on-shelf-stock-availability-ox04t\data.yaml', epochs=1, imgsz=416)
# model.export(format="onnx")