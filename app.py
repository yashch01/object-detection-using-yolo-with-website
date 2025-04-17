from flask import Flask, current_app
import cv2
import numpy as np
from modules.image_detection import image_bp
from modules.video_detection import video_bp
from modules.webcam_detection import webcam_bp
from modules.main import main_bp
import os

app = Flask(__name__, template_folder='templates')

def load_yolo():
    print("Loading YOLO model...")
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print("YOLO model loaded.")
    return net, classes, output_layers

@app.before_request
def initialize():
    with app.app_context():
        if not hasattr(current_app, 'yolo_loaded'):
            net, classes, output_layers = load_yolo()
            current_app.net = net
            current_app.classes = classes
            current_app.output_layers = output_layers
            current_app.yolo_loaded = True
            print("YOLO model stored in current_app.")
        else:
            print("YOLO model already loaded in current_app.")

# Register Blueprints
app.register_blueprint(main_bp)
app.register_blueprint(image_bp)
app.register_blueprint(video_bp)
app.register_blueprint(webcam_bp)

if __name__ == '__main__':
    app.run(debug=True)