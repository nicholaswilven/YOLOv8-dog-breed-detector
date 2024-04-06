from ultralytics import YOLO
import cv2
import os

# Path to YOLOv5 model weights
model_path = "runs/detect/train/weights/best.pt"
# Input video file
infer_filename = "your_file.mp4"

# Initialize YOLO model
model = YOLO(model_path)
# Path to input video
video_path = "data/" + infer_filename
# Path to output video
output_video_path = "data/result/" + infer_filename
# Open input video
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4')  # Codec for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Process each frame and draw bounding boxes
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection on the frame
    results = model(frame)
    
    # Extract bounding box coordinates and class predictions
    boxes = results[0].boxes.numpy()
    class_ids = results[0].names
    for box in boxes:
        xmin, ymin, xmax, ymax = [int(x) for x in box.xyxy[0]]
        class_id = int(box.cls[0])
        print(xmin, ymin, xmax, ymax, class_id)
        
        # Draw bounding box on the frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Draw class label
        class_name = class_ids[class_id]
        cv2.putText(frame, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Write the frame with bounding boxes to the output video
    out.write(frame)

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()