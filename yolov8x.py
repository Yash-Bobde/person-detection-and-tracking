import cv2
from ultralytics import YOLO
import ultralytics
print(ultralytics.__version__)

# Load the YOLOv8 model (pre-trained on COCO dataset)
model = YOLO('yolov8x.pt')  # Adjust model size if needed (e.g., yolov8s.pt for better accuracy)

def count_and_display_people(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_num = 0

    # Processing Each frame in the video.
    # This loop reads the video frame-by-frame.
    while cap.isOpened():
        ret, frame = cap.read()
        # If ret is False, the video has ended, and the loop breaks.
        if not ret:
            break

        # Run YOLO inference (only 'person' class with ID 0)
        results = model(frame, classes=[0])

        # Track the number of people detected in the current frame
        current_frame_people = 0

        # Process Detected Objects and draw bounding boxes
        for result in results:            
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Check if class ID is 'person'
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    person_id = (int(x1), int(y1), frame_num)
                    
                    current_frame_people += 1

                    # Draw bounding box and label
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # current frame people count on the frame
        frame_count_text = f"People in Frame: {current_frame_people}"
        cv2.putText(frame, frame_count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("YOLOv8 People Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'videos/1.mp4'  # Replace with the path to your video
# count_and_display_people(video_path)
