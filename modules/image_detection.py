from flask import Blueprint, render_template, request, current_app
import base64
import os
import io
from PIL import Image
import cv2
import numpy as np

image_bp = Blueprint('image_detection', __name__, url_prefix='/')

def perform_image_detection(image_path):
    net = current_app.net
    classes = current_app.classes
    output_layers = current_app.output_layers
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    detections = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            detections.append({"label": label, "confidence": f"{confidence:.2f}", "box": [x, y, w, h]})
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label}: {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    img_buffer = io.BytesIO()
    cv2.imwrite("temp_detected.jpg", image)
    with open("temp_detected.jpg", "rb") as f:
        img_buffer.write(f.read())
    os.remove("temp_detected.jpg")
    img_buffer.seek(0)

    detection_details = ""
    for det in detections:
        detection_details += f"Class: {det['label']}, Confidence: {det['confidence']}, Box: {det['box']}\n"

    return img_buffer.getvalue(), detection_details

@image_bp.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image part in the request."
    file = request.files['image']
    if file.filename == '':
        return "No selected image."
    if file:
        image_path = os.path.join("temp", file.filename)
        file.save(image_path)
        detected_image_bytes, detection_details = perform_image_detection(image_path)
        os.remove(image_path)

        return render_template('result.html', detected_image_bytes=base64.b64encode(detected_image_bytes).decode('utf-8'), detection_details=detection_details, input_type='image', original_filename=file.filename)
    return "Something went wrong with the upload."