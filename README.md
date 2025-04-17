# YOLOv3 Object Detection Web Application

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23white.svg?style=for-the-badge&logo=opencv&logoColor=black)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logo=PIL&logoColor=white)](https://pillow.readthedocs.io/en/stable/)


This web application leverages the YOLOv3 object detection model to identify objects in real-time from a webcam feed, as well as in uploaded images and videos. Built with Flask, it provides an intuitive interface for users to interact with the object detection capabilities. This project is designed for deployment on platforms like Vercel, making it easily accessible online.

## Features

* **Real-time Webcam Object Detection:** Detect objects in your live webcam feed with bounding boxes and confidence scores displayed directly in your browser.
* **Image Object Detection:** Upload static image files (e.g., JPG, PNG) and see the detected objects highlighted.
* **Video Object Detection:** Upload video files (e.g., MP4) for object detection. The application processes the video and provides detection results.
* **Clear Output:** Detected objects are labeled with their class names and confidence percentages.
* **User-Friendly Interface:** Simple and intuitive web interface built with HTML, CSS, and JavaScript.
* **Theme Toggle:** Includes a light/dark theme toggle for user preference.

## Installation

### Prerequisites

* **Python 3.x:** Ensure you have Python 3 installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
* **pip:** Python package installer, usually included with Python.
* **Git:** For cloning the repository. You can download it from [git-scm.com](https://git-scm.com/downloads).
* **(Optional) Virtual Environment:** Recommended for isolating project dependencies.

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    ```
    (Replace `<repository_url>` with the actual URL of your GitHub repository)

2.  **Navigate to the Project Directory:**
    ```bash
    cd yolo_web_app
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv  # For macOS and Linux
    # OR
    python -m venv venv   # For older Python versions
    # OR
    py -m venv venv      # On Windows
    ```

4.  **Activate the Virtual Environment:**
    ```bash
    source venv/bin/activate  # On macOS and Linux
    venv\Scripts\activate   # On Windows
    ```

5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure you have generated the `requirements.txt` file as per the previous instructions.)

## Usage

### Running Locally

1.  After completing the installation steps, navigate to the root directory of the project in your terminal.
2.  Run the Flask development server:
    ```bash
    python app.py
    ```
3.  Open your web browser and go to `http://127.0.0.1:5000/`.
4.  You will see the main page with links to the different detection features.

### Using the Application

* **Webcam Detection:**
    1.  Click on the "Webcam Detection" link.
    2.  Click the "Start Webcam" button to access your camera (you'll be prompted for permission).
    3.  Object detection will run in real-time, displaying bounding boxes and labels on the video feed.
    4.  You can start and stop the recording, and download the recorded video (which includes the detections).

* **Image Upload:**
    1.  Click on the "Image Detection" link on the main page.
    2.  Click the "Choose File" button to select an image from your computer.
    3.  Click the "Upload Image" button.
    4.  The detected image with bounding boxes and labels will be displayed on the results page.

* **Video Upload:**
    1.  Click on the "Video Detection" link on the main page.
    2.  Click the "Choose File" button to select a video file from your computer.
    3.  Click the "Upload Video" button.
    4.  The application will process the video, and the detected video (or detection details, depending on the final implementation for Vercel) will be shown on the results page.

## Dependencies

* **Flask:** A lightweight Python web framework for building the web application.
* **OpenCV-Python (`cv2`):** A library for computer vision tasks, used here for image and video processing and object detection.
* **NumPy:** A library for numerical computing in Python, essential for working with image data.
* **Pillow (`PIL`):** The Python Imaging Library, used for opening and manipulating image files.
* **Gunicorn:** A WSGI HTTP server for deploying Python web applications (used for Vercel).

## YOLOv3 Model

This application utilizes the YOLOv3 object detection model. The following files are required:

* `yolov3.cfg`: Contains the architectural configuration of the YOLOv3 model.
* `yolov3.weights`: Contains the pre-trained weights of the YOLOv3 model, learned from the COCO dataset.
* `coco.names`: A text file listing the names of the 80 object classes that the YOLOv3 model can detect.

**Note on `yolov3.weights`:** It is currently ignored under .gitignore due to its larger size.

## Potential Future Enhancements

* Implement object tracking to follow detected objects across frames.
* Allow users to select different YOLO models.
* Add more detailed information about the detected objects.
* Improve the user interface and styling.
* Implement user authentication.

## Contributing

(Optional) If you would like others to contribute to your project, you can add guidelines here on how to submit pull requests, report issues, etc.


## Acknowledgments

* This project utilizes the pre-trained YOLOv3 model.
* Thanks to the Flask, OpenCV, NumPy, and Pillow teams for their excellent libraries.