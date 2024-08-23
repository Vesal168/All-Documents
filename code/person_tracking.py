
import os
import torch
import torchvision
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

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

# Define the Kalman filter for tracking
class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.dt = dt  # time step
        self.u = np.array([[u_x], [u_y], [0], [0]])  # motion vector with vx, vy, ax, ay
        self.std_acc = std_acc  # acceleration
        # Process matrix
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        # Measurement noise covariance
        self.R = np.array([[x_std_meas**2, 0],
                           [0, y_std_meas**2]])
        # Process noise covariance
        self.Q = np.array([[self.dt**4/4, 0, self.dt**3/2, 0],
                           [0, self.dt**4/4, 0, self.dt**3/2],
                           [self.dt**3/2, 0, self.dt**2, 0],
                           [0, self.dt**3/2, 0, self.dt**2]]) * std_acc**2
        # Estimate error covariance
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        # Prediction
        self.x = np.dot(self.A, self.x) + self.u
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        # Update
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

# Calculate centroid of a bounding box
def calculate_centroid(box):
    return [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]

# Define the function to perform human detection and tracking on a video
def detect_and_track(video_path, output_dir):
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, video_name)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    # Initialize Kalman filters for each detected human
    kalman_filters = []

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

        # Initialize detected indices
        detected_indices = []

        # Create centroids of detected bounding boxes
        detected_centroids = [calculate_centroid(box.detach().numpy().astype(int)) for box in human_boxes]

        # Assign detected bounding boxes to Kalman filters using Hungarian Algorithm
        # if kalman_filters:
        #     kalman_centroids = [kalman_filter.x[:2].flatten().astype(int).tolist() for kalman_filter in kalman_filters]
        #     cost_matrix = np.array([[np.linalg.norm(np.array(detected_centroid) - np.array(kalman_centroid)) for detected_centroid in detected_centroids] for kalman_centroid in kalman_centroids])
        #     kalman_indices, detected_indices = linear_sum_assignment(cost_matrix)
        # Assign detected bounding boxes to Kalman filters using Hungarian Algorithm
        if kalman_filters:
            kalman_centroids = [kalman_filter.x[:2].flatten().astype(int).tolist() for kalman_filter in kalman_filters]
            detected_centroids = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in human_boxes]
            cost_matrix = np.array([[np.linalg.norm(np.array(detected_centroid) - np.array(kalman_centroid)) for detected_centroid in detected_centroids] for kalman_centroid in kalman_centroids])
            kalman_indices, detected_indices = linear_sum_assignment(cost_matrix)



            # Update Kalman filters with assigned bounding boxes
            for kalman_index, detected_index in zip(kalman_indices, detected_indices):
                kalman_filters[kalman_index].update(np.array(detected_centroids[detected_index]))

        # Create new Kalman filters for unassigned bounding boxes
        for i, box in enumerate(human_boxes):
            if i not in detected_indices:
                box = box.detach().numpy().astype(int)
                kalman_filter = KalmanFilter(dt=1, u_x=0, u_y=0, std_acc=0.2, x_std_meas=2, y_std_meas=2)
                kalman_filter.x = np.array([[box[0] + (box[2] - box[0]) / 2], [box[1] + (box[3] - box[1]) / 2], [0], [0]])  # Initial state
                kalman_filters.append(kalman_filter)

        # Draw bounding boxes and predicted states on frame
        for kalman_filter in kalman_filters:
            predicted_state = kalman_filter.x.astype(int)
            cv2.circle(frame, (predicted_state[0][0], predicted_state[1][0]), 4, (255, 0, 0), -1)

        # Write the frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Input and output directories
input_dir = r'E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\dataset\train/'
output_dir = r'E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\dataset\output_person_detection_and_tracking'

# Iterate through all video files in the input directory
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.avi'):
            video_path = os.path.join(root, file)
            detect_and_track(video_path, output_dir)





