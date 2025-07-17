import onnxruntime
import numpy as np
import cv2

# Load model
session = onnxruntime.InferenceSession("../models/rambuid.onnx", providers=["CPUExecutionProvider"])

# Ambil input dan output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load gambar dan pre-process sesuai model YOLO
img = cv2.imread("../test.jpg")
img = cv2.resize(img, (640, 640))
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR ke RGB dan ke CHW
img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0  # Normalize

# Inference
outputs = session.run([output_name], {input_name: img})[0]

print(outputs)
