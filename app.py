from __future__ import annotations

import base64
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from detector import ProCVDetector


app = Flask(__name__)
detector = ProCVDetector()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})


@app.post("/api/detect")
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Missing image upload."}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"error": "Empty image upload."}), 400

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Unsupported image format."}), 400

    detection, all_detections = detector.detect_frame(frame)
    annotated = detector.annotate_frame(frame, detection)
    image_base64 = base64.b64encode(detector.encode_jpeg(annotated)).decode("ascii")

    payload = {
        "detection": detection.to_dict() if detection else None,
        "also_seen": [item.to_dict() for item in all_detections[:5]],
        "annotated_image": f"data:image/jpeg;base64,{image_base64}",
    }
    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
