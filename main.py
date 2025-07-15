from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n.pt")  # or any other model like yolov8s.pt

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)

    # Draw boxes on image for labeled_image_base64 (optional)
    annotated_img = img.copy()
    annotations = []
    for det in results[0].boxes:
        cls_id = int(det.cls[0])
        label = model.names[cls_id]
        conf = float(det.conf[0])
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        annotations.append({
            "label": label,
            "confidence": round(conf, 5),
            "bounding_box": {
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2
            }
        })
        # Draw box (optional)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Encode image to base64
    _, buffer = cv2.imencode(".jpg", annotated_img)
    labeled_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "image_id": filename,
        "annotations": annotations,
        "labeled_image_base64": labeled_base64
    }


