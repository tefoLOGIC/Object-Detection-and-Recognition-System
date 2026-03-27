import cv2
import numpy as np
import time
import os
from datetime import datetime

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load Yolo
weights_path = os.path.join(current_dir, "YOLOv3 MODEL FILES", "yolov3.weights")
config_path = os.path.join(current_dir, "YOLOv3 MODEL FILES", "yolov3.cfg")
names_path = os.path.join(current_dir, "YOLOv3 MODEL FILES", "coco.names")

yolo = cv2.dnn.readNet(weights_path, config_path)


classes = []
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Constants for video output
WIDTH = 1000
HEIGHT = 1080
OUTPUT_FOLDER = "OUTPUT VIDEOS"

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Function to generate output video filename
def generate_output_filename():
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(OUTPUT_FOLDER, f'output_video_{current_datetime}.avi')

# Loading video
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
frame_id = 0

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = generate_output_filename()
output_video = cv2.VideoWriter(output_path, fourcc, 20.0, (WIDTH, HEIGHT))

try:
    while True:
        _, frame = cap.read()
        frame_id += 1

        height, width, _ = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        yolo.setInput(blob)
        outs = yolo.forward(output_layers)

        # Showing informations on the screen

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, color, 3)

        # Resize the frame to specified width and height
        resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # Write the frame into the video file
        output_video.write(resized_frame)

        cv2.imshow("result", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

        # Save the video and create a new one for every run
        if frame_id % 500 == 0:  # Change this number based on the number of frames you want in each video
            output_video.release()  # Release the current video
            output_path = generate_output_filename()  # Generate a new filename
            output_video = cv2.VideoWriter(output_path, fourcc, 20.0, (WIDTH, HEIGHT))  # Create a new video file

except KeyboardInterrupt:
    # Release everything if job is finished
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
