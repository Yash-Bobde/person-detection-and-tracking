import cv2
import numpy as np  # Import numpy for array manipulation
from ultralytics import YOLO
from sort import Sort  # Import SORT tracker https://github.com/abewley/sort

# Initialize the YOLO model
model = YOLO('yolov8x.pt')  # Replace with the correct model if needed

# Initialize SORT tracker
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Set to store unique person IDs
unique_person_ids = set()

def count_and_display_people(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_num = 0  # Track the frame number

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if the video ends

        # Run YOLO inference (only 'person' class with ID 0)
        results = model(frame, classes=[0])

        # Collect detections in [x1, y1, x2, y2, confidence] format
        detections = []

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Class ID 0 corresponds to 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = box.conf[0].item()  # Get confidence score
                    detections.append([x1, y1, x2, y2, confidence])

        # Convert detections to a NumPy array (or an empty array if no detections)
        detections = np.array(detections) if detections else np.empty((0, 5))

        # Update tracker with the current frame's detections
        tracked_objects = tracker.update(detections)

        current_frame_people = 0  # Count people in the current frame

        # Process tracked objects
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)

            # If the ID is new, add it to the unique set
            if obj_id not in unique_person_ids:
                unique_person_ids.add(obj_id)

            # Draw bounding box and label with the unique ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {obj_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            current_frame_people += 1

        # Display people count for the frame and total unique people
        frame_text = f"People in Frame: {current_frame_people}"
        unique_count_text = f"Total Unique People: {len(unique_person_ids)}"
        cv2.putText(frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, unique_count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("YOLO + SORT Tracking", frame)

        # Press 'q' to quit the video early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total unique people detected in the video: {len(unique_person_ids)}")
    
# Example usage
video_path = 'videos/1.mp4'  # Replace with your video path
count_and_display_people(video_path)
