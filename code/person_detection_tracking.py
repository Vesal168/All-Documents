import os
import torch
import torchvision
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import time

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load the COCO class labels
CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

#Define a tracked object class
class TrackedObject:
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                               [0, 1, 0, 1],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]])
        self.kf.P *= 1000  # initial uncertainty
        self.kf.R = np.diag([0.5, 0.5])  # measurement uncertainty

        # Initialize state vector with bbox center
        x, y, w, h = bbox
        self.kf.x = np.array([x + w / 2, y + h / 2, 0, 0]).reshape(-1, 1)

        self.history = []

    def predict(self):
        self.kf.predict()

    def update(self, bbox):
        x, y, w, h = bbox
        z = np.array([x + w / 2, y + h / 2]).reshape(-1, 1)
        self.kf.update(z)

        # Save predicted state for drawing
        self.history.append(self.kf.x.flatten())

# Hungarian algorithm for data association
def hungarian_algorithm(distances):
    row_ind, col_ind = linear_sum_assignment(distances)
    return row_ind, col_ind

# Define the function to perform human detection and tracking on a video
def detect_and_track(video_path, output_dir):
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, video_name)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    tracked_objects = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_tensor = torchvision.transforms.functional.to_tensor(frame)
        outputs = model([image_tensor])

        # Get the bounding boxes and labels for human detections
        boxes = outputs[0]['boxes']
        labels = outputs[0]['labels']

        # Filter out detections where the label corresponds to 'person'
        human_boxes = [boxes[i] for i in range(len(boxes)) if labels[i] == 1]

        # Initialize or update tracked objects
        if not tracked_objects:
            for box in human_boxes:
                # Detach the tensor and convert it to a NumPy array
                box_np = box.detach().numpy()
                tracked_objects.append(TrackedObject(box_np))

        else:
            # Predict object states
            for obj in tracked_objects:
                obj.predict()

        # Draw bounding boxes around tracked objects
        for obj in tracked_objects:
            for state in obj.history:
                x, y = map(int, state[:2])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Write the frame with tracked objects to the output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

input_dir = r'E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\dataset\test/'
output_dir = r'E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\dataset\output_person_detection_and_tracking'

# Iterate through all video files in the input directory
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.avi'):
            video_path = os.path.join(root, file)
            detect_and_track(video_path, output_dir)



            
