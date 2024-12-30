# Object Detection and Tracking

This repository contains a Python script for object detection and tracking using the YOLOv5 model and OpenCV. You can use this script with any video file to track objects effectively. 

## Files in the Repository

- `object_detection_tracking.py`: The main Python script for object detection and tracking.
- `Untitled video - Made with Clipchamp.mp4`: A sample video file for testing the script.
- `yolov5s.pt`: The YOLOv5 small model weights file.

## How to Use

### Prerequisites

1. Install Python 3.8+.
2. Install the required dependencies by running:

   ```bash
   pip install -r requirements.txt
Ensure the following libraries are included:

-torch
-opencv-python
-ultralytics

Running the Script
Ensure the video file you want to use for tracking is in the same directory as the script.

Update the video_path variable in object_detection_tracking.py if needed, or replace the existing video file with your own.

Run the script in your terminal using:

bash

python object_detection_tracking.py
Press q at any time to stop the video playback.

Features
Tracks multiple objects using YOLOv5 detection and CSRT trackers.
Displays real-time bounding boxes and labels for detected objects.
Displays the current FPS in the video playback window.

Customization
To use a different video, replace Untitled video - Made with Clipchamp.mp4 with your video file.
You can adjust the object detection confidence threshold in the detect_and_track function.
Modify DETECTION_INTERVAL to control how often YOLOv5 runs (default: every 10 frames).

Notes
Ensure your system has a GPU if you want optimal performance.
The YOLOv5 model is loaded using torch.hub and will download the required files if not already available.

License
This project is open-source and available under the MIT License.
