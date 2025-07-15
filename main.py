from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import base64

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform detection
    results = model(img)
    result = results[0]

    # Extract annotations
    annotations = []
    for box in result.boxes:
        b = box.xyxy[0].tolist()
        cls = int(box.cls[0].item())
        label = result.names[cls]
        conf = float(box.conf[0].item())
        annotations.append({
            "label": label,
            "confidence": round(conf, 4),
            "bounding_box": {
                "x_min": round(b[0], 2),
                "y_min": round(b[1], 2),
                "x_max": round(b[2], 2),
                "y_max": round(b[3], 2)
            }
        })

    # Draw boxes on image for preview
    result_plotted = result.plot()
    _, buffer = cv2.imencode('.jpg', result_plotted)
    base64_img = base64.b64encode(buffer).decode('utf-8')

    return {
        "image_id": file.filename,
        "annotations": annotations,
        "labeled_image_base64": base64_img
    }

