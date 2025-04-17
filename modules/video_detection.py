from flask import Blueprint, render_template, request, current_app
import base64
import os
import cv2
import numpy as np

video_bp = Blueprint('video_detection', __name__, url_prefix='/')

def perform_video_detection(video_path):
    net = current_app.net
    classes = current_app.classes
    output_layers = current_app.output_layers
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join("temp", f"{video_name}_detected.mp4")
    output_text_path = os.path.join("temp", f"{video_name}_detection_results.txt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None, None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    f = open(output_text_path, "w")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        height, width, channels = frame.shape
        frame_count += 1
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        boxes, confidences, class_ids = [], [], []
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
                f.write(f"Frame {frame_count}: Class: {label}, Confidence: {confidence:.2f}, Box: [{x}, {y}, {w}, {h}]\n")
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label}: {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        out.write(frame)
    cap.release()
    out.release()
    f.close()

    try:
        with open(output_video_path, 'rb') as video_file:
            video_bytes = video_file.read()
        os.remove(output_video_path)
    except FileNotFoundError:
        return None, None

    try:
        with open(output_text_path, 'r') as text_file:
            detection_details = text_file.read()
        os.remove(output_text_path)
    except FileNotFoundError:
        detection_details = ""

    return video_bytes, detection_details

@video_bp.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No video part in the request."
    file = request.files['video']
    if file.filename == '':
        return "No selected video."
    if file:
        video_path = os.path.join("temp", file.filename) # Keep 'data' for video upload for now
        file.save(video_path)
        video_bytes, detection_details = perform_video_detection(video_path)
        os.remove(video_path)

        if video_bytes and detection_details is not None:
            return render_template('result.html', video_bytes=base64.b64encode(video_bytes).decode('utf-8'), detection_details=detection_details, input_type='video', original_filename=file.filename)
        else:
            return "Error processing video."
    return "Something went wrong with the video upload."