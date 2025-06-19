from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image, ImageOps
import base64
import io
import os
import cv2

from ultralytics import YOLO

# ⚠️ Assicurati che il modello sia nel path corretto
model = YOLO("best-7.pt")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def prepare_image(file):
    image_original = Image.open(file).convert("RGB")
    image_original = ImageOps.exif_transpose(image_original)  # ✅ corregge la rotazione
    image_resized = image_original.resize((384, 384))         # ✅ inferenza più veloce
    return image_original, image_resized

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Nessun file ricevuto"}), 400

    file = request.files["image"]
    print(f"Ricevuto file: {file.filename}")

    try:
        image_original, image_resized = prepare_image(file)
    except Exception as e:
        print(f"Errore nel caricamento immagine: {e}")
        return jsonify({"error": "Impossibile processare immagine"}), 400

    print("Inferenza in corso")
    results = model(image_resized)[0]
    print("Inferenza completata")

    boxes = results.boxes
    class_names = model.names
    detected_objects = []

    orig_w, orig_h = image_original.size
    resized_w, resized_h = image_resized.size

    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    draw = np.array(image_original)

    for box in boxes:
        confidence = float(box.conf.item())
        if confidence < 0.5:
            continue

        cls_id = int(box.cls.item())
        class_name = class_names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # ✅ Scalo per immagine originale
        x1_orig = int(x1 * scale_x)
        y1_orig = int(y1 * scale_y)
        x2_orig = int(x2 * scale_x)
        y2_orig = int(y2 * scale_y)

        # ✅ Bounding box normalizzate
        bbox = [
            x1_orig / orig_w,
            y1_orig / orig_h,
            x2_orig / orig_w,
            y2_orig / orig_h
        ]

        detected_objects.append({
            "class": class_name,
            "confidence": confidence,
            "bbox": bbox
        })

        # ✅ Disegna box
        cv2.rectangle(draw, (x1_orig, y1_orig), (x2_orig, y2_orig), (255, 0, 0), 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(draw, label, (x1_orig, y1_orig - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    draw = Image.fromarray(draw)

    buffer = io.BytesIO()
    draw.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({
        "processed_image_base64": f"data:image/jpeg;base64,{img_str}",
        "detected_objects": detected_objects,
        "message": "Rilevamento completato!"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
