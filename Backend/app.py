from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import base64
import io
from ultralytics import YOLO

# ⚠️ Assicurati che best.pt sia nella stessa cartella o metti il path giusto
model = YOLO("best-7.pt")

app = Flask(__name__)
CORS(app)

def prepare_image(file):
    image = Image.open(file).convert("RGB")
    return image
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "awake"}), 200

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Nessun file ricevuto"}), 400

    file = request.files["image"]
    image = prepare_image(file)

    results = model(image)[0]
    boxes = results.boxes
    class_names = model.names

    detected_objects = []
    image_width, image_height = image.size

    for box in boxes:
        confidence = float(box.conf.item())
        if confidence < 0.5:
            continue

        cls_id = int(box.cls.item())
        class_name = class_names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        bbox = [
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height
        ]

        detected_objects.append({
            "class": class_name,
            "confidence": confidence,
            "bbox": bbox
        })

    # Disegna box sull'immagine
    try:
        import cv2
        draw = np.array(image)
        for box in boxes:
            confidence = float(box.conf.item())
            if confidence < 0.5:
                continue

            cls_id = int(box.cls.item())
            class_name = class_names[cls_id]
            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]

            cv2.rectangle(draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(draw, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        draw = Image.fromarray(draw)
    except:
        draw = image

    buffer = io.BytesIO()
    draw.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({
        "processed_image_base64": f"data:image/jpeg;base64,{img_str}",
        "detected_objects": detected_objects,
        "message": "Rilevamento completato!"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
