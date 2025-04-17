from flask import Blueprint, render_template, request, jsonify, current_app
import base64
import base64
import cv2
import numpy as np
import io
from PIL import Image

webcam_bp = Blueprint('webcam_detection', __name__, url_prefix='/')

@webcam_bp.route('/webcam')
def webcam_page():
    return render_template('webcam.html')

@webcam_bp.route('/process_frame_webcam', methods=['POST'])
def process_frame_webcam():
    net = current_app.net
    classes = current_app.classes
    output_layers = current_app.output_layers
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    detections = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                # Handle potential Infinity or NaN values
                if not np.isfinite(confidence):
                    confidence = 0.0  # Or some other appropriate default value
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)  # Increased confidence threshold
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            detections.append({"label": label, "confidence": confidence, "box": [x, y, w, h]})

    return jsonify({'detections': detections})