import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model (pre-trained on COCO dataset)
model = YOLO('yolov8x.pt')  # You can also try yolov8s.pt for better accuracy

def iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    # Calculate intersection
    inter_x1, inter_y1 = max(x1, x1b), max(y1, y1b)
    inter_x2, inter_y2 = min(x2, x2b), min(y2, y2b)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def track_and_count_people(video_path, iou_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    unique_people = []  # List to store bounding boxes of unique people
    person_id = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference (only 'person' class with ID 0)
        results = model(frame, classes=[0])

        # Process detections and update tracking list
        current_frame_boxes = []

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Check if class is 'person'
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    current_frame_boxes.append((x1, y1, x2, y2))

                    # Check if this person is new (no high IoU with existing ones)
                    is_new_person = True
                    for unique_box in unique_people:
                        if iou((x1, y1, x2, y2), unique_box) > iou_threshold:
                            is_new_person = False
                            break

                    if is_new_person:
                        unique_people.append((x1, y1, x2, y2))
                        person_id += 1  # Increment person count

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display total unique people count on the frame
        total_count_text = f"Total Unique People: {len(unique_people)}"
        cv2.putText(frame, total_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("YOLOv8 People Detection with Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total unique people detected in the video: {len(unique_people)}")

# Example usage
video_path = 'videos/1.mp4'  # Replace with your video path
track_and_count_people(video_path)
