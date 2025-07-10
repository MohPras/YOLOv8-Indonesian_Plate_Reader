from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Saat produksi ganti dengan domainmu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once saat server start
model_plat = YOLO("Model_object_deteksi/best.pt")
model_char = YOLO("Model_caracter_deteksi/best.pt")

@app.post("/detect-plat")
async def detect_plate(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    plate_number_final = ""
    boxes_output = []

    # Deteksi plat
    results_plat = model_plat.predict(source=img, conf=0.25, save=False)
    for box in results_plat[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        boxes_output.append({"label": "plat", "x1": x1, "y1": y1, "x2": x2, "y2": y2})

        plat_crop = img[y1:y2, x1:x2]
        results_char = model_char.predict(plat_crop, conf=0.5, save=False)

        detected_chars = []
        for char_box, char_cls, char_conf in zip(
            results_char[0].boxes.xyxy.cpu().numpy(),
            results_char[0].boxes.cls.cpu().numpy(),
            results_char[0].boxes.conf.cpu().numpy()
        ):
            cx1, cy1, cx2, cy2 = char_box
            cx1_orig = int(cx1 + x1)
            cy1_orig = int(cy1 + y1)
            cx2_orig = int(cx2 + x1)
            cy2_orig = int(cy2 + y1)

            char_label = results_char[0].names[int(char_cls)]
            detected_chars.append({"char": char_label, "x": cx1_orig, "y": cy1_orig})

            boxes_output.append({
                "label": "char",
                "char": char_label,
                "x1": cx1_orig,
                "y1": cy1_orig,
                "x2": cx2_orig,
                "y2": cy2_orig
            })

        # Susun karakter dari kiri ke kanan
        detected_chars_sorted = sorted(detected_chars, key=lambda k: k['x'])
        plate_number_final = "".join([d['char'] for d in detected_chars_sorted])

        break  # Asumsikan hanya 1 plat

    return JSONResponse(content={
        "plate_number": plate_number_final,
        "boxes": boxes_output
    })
    
# install
# uvicorn main:app --reload