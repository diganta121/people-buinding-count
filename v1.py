import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Make sure you have sort.py in your directory or installed

# Define the polygon zones (manually defined based on entrance area)
entry_zone = np.array([[200, 300], [400, 300], [400, 500], [200, 500]])
exit_zone = np.array([[200, 500], [400, 500], [400, 700], [200, 700]])

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize SORT tracker
tracker = Sort()

# Dictionary to store previous positions of tracked people
track_memory = {}

# Counter
entered = 0
exited = 0


def is_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


# Open camera/video
cap = cv2.VideoCapture("video.mp4")  # or use 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    # Filter people only (class 0 = person)
    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = result
        if int(cls) == 0 and score > 0.4:
            detections.append([x1, y1, x2, y2, score])

    # Track people
    tracked = tracker.update(np.array(detections))

    # Draw zones
    cv2.polylines(frame, [entry_zone], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(frame, [exit_zone], isClosed=True, color=(0, 0, 255), thickness=2)

    # Check each tracked object
    for person in tracked:
        x1, y1, x2, y2, track_id = map(int, person[:5])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        current_point = (cx, cy)
        if track_id in track_memory:
            prev_point = track_memory[track_id]
            if is_inside_polygon(prev_point, entry_zone) and is_inside_polygon(
                current_point, exit_zone
            ):
                exited += 1
                print(f"Person {track_id} exited")
            elif is_inside_polygon(prev_point, exit_zone) and is_inside_polygon(
                current_point, entry_zone
            ):
                entered += 1
                print(f"Person {track_id} entered")

        track_memory[track_id] = current_point

        # Draw box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )

    # Display count
    cv2.putText(
        frame,
        f"In: {entered} | Out: {exited} | Inside: {entered - exited}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Entrance Monitor", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
