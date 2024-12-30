import cv2
import torch
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to process detections and initialize trackers
def detect_and_track(frame):
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # x_min, y_min, x_max, y_max, confidence, class_id

    trackers = []
    tracker_types = []

    for detection in detections:
        x_min, y_min, x_max, y_max, confidence, class_id = detection
        if confidence > 0.3:  # Adjust threshold as needed
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

            # Initialize tracker
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, bbox)

            trackers.append(tracker)
            tracker_types.append(int(class_id))

            # Debug: Display bounding box details
            print(f"Tracker initialized for class: {model.names[int(class_id)]}, BBox: {bbox}")

    return trackers, tracker_types

# Hardcoded video file path (ensure the file is in the same directory)
video_path = "Untitled video - Made with Clipchamp.mp4"

# Start video capture
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video source {video_path}.")
    exit()

# Initialize variables
prev_frame_time = 0
frame_counter = 0
DETECTION_INTERVAL = 10

trackers = []
tracker_types = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended or failed to read frame.")
        break

    frame_resized = cv2.resize(frame, (640, 480))  # Consistent resizing for processing

    if frame_counter % DETECTION_INTERVAL == 0:
        # Run YOLO detection
        trackers, tracker_types = detect_and_track(frame_resized)
    else:
        # Update trackers
        new_trackers = []
        new_tracker_types = []
        for i, tracker in enumerate(trackers):
            success, bbox = tracker.update(frame_resized)
            if success:
                x, y, w, h = bbox
                cv2.rectangle(frame_resized, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                label = f"{model.names[tracker_types[i]]}"
                cv2.putText(frame_resized, label, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                new_trackers.append(tracker)
                new_tracker_types.append(tracker_types[i])
        trackers = new_trackers
        tracker_types = new_tracker_types

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display frame
    cv2.imshow("CCTV Object Detection and Tracking", frame_resized)

    frame_counter += 1

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
