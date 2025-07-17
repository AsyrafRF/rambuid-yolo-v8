from ultralytics import YOLO

# Load model .pt kamu
model = YOLO("../models/rambuid.pt")

# Export ke ONNX
model.export(format="onnx")
