
# child own dataset

# import torch
# import torchvision
# import cv2

# # Load the pre-trained Faster R-CNN model
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()

# # Load the COCO class labels
# CLASSES = [
#     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
#     'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
#     'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
#     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]

# # Define the function to perform human detection on a video
# def detect_human(video_path):
#     cap = cv2.VideoCapture(video_path)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         image_tensor = torchvision.transforms.functional.to_tensor(frame)
#         outputs = model([image_tensor])

#         # Get the bounding boxes and labels for human detections
#         boxes = outputs[0]['boxes']
#         labels = outputs[0]['labels']

#         # Filter out detections where the label corresponds to 'person'
#         human_boxes = [boxes[i] for i in range(len(boxes)) if labels[i] == 1]

#         # Draw bounding boxes around detected humans
#         for box in human_boxes:
#             box = box.detach().numpy().astype(int)
#             cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

#         # Display the video frame with bounding boxes
#         cv2.imshow('Person Detection', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Test the human detection function on a video
# video_path = r'/Users/visal/Documents/Human-Human-Interaction/asset/child.mp4'
# detect_human(video_path)


# import os
# import torch
# import torchvision
# import cv2
# import onnx
# from onnx import shape_inference

# # Load the pre-trained Faster R-CNN model
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()

# # Define paths for saving the model
# model_save_dir = r'E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\src\model'
# frozen_graph_path = os.path.join(model_save_dir, 'frozen_inference_graph.pb')
# graph_txt_path = os.path.join(model_save_dir, 'graph.pbtxt')

# # Save the model as a pre-trained Frozen Graph
# input_names = ['image']
# output_names = ['boxes', 'labels', 'scores']
# torch.onnx.export(model, torch.rand(1, 3, 300, 300), frozen_graph_path, input_names=input_names, output_names=output_names, opset_version=11)

# # Generate a graph text file for visualization
# onnx_model = onnx.load(frozen_graph_path)
# onnx_model = shape_inference.infer_shapes(onnx_model)
# onnx.checker.check_model(onnx_model)
# with open(graph_txt_path, 'w') as f:
#     f.write(str(onnx_model.graph))

# # Define the function to perform human detection on a video
# def detect_human(video_path, output_dir):
#     video_name = os.path.basename(video_path)
#     output_path = os.path.join(output_dir, video_name)
    
#     cap = cv2.VideoCapture(video_path)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         image_tensor = torchvision.transforms.functional.to_tensor(frame)
#         outputs = model([image_tensor])

#         # Get the bounding boxes and labels for human detections
#         boxes = outputs[0]['boxes']
#         labels = outputs[0]['labels']

#         # Filter out detections where the label corresponds to 'person'
#         human_boxes = [boxes[i] for i in range(len(boxes)) if labels[i] == 1]

#         # Draw bounding boxes around detected humans
#         for box in human_boxes:
#             box = box.detach().cpu().numpy().astype(int)
#             cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

#         # Write the frame with bounding boxes to the output video
#         out.write(frame)

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# # Define the paths for input and output directories
# input_dir = r'E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\dataset\train'
# output_dir = r'E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\dataset\test1'

# # Iterate through all video files in the input directory
# for root, dirs, files in os.walk(input_dir):
#     for file in files:
#         if file.endswith('.avi'):
#             video_path = os.path.join(root, file)
#             detect_human(video_path, output_dir)


import os
import torch
import torchvision
import cv2
import onnx
from onnx import shape_inference

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define paths for saving the model
model_save_dir = r'E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\src\model'
frozen_graph_path = os.path.join(model_save_dir, 'frozen_inference_graph.pb')
graph_txt_path = os.path.join(model_save_dir, 'graph.pbtxt')

# Save the model as a pre-trained Frozen Graph
input_names = ['image']
output_names = ['boxes', 'labels', 'scores']
torch.onnx.export(model, torch.rand(1, 3, 300, 300), frozen_graph_path, input_names=input_names, output_names=output_names, opset_version=11)

# Generate a graph text file for visualization
onnx_model = onnx.load(frozen_graph_path)
onnx_model = shape_inference.infer_shapes(onnx_model)
onnx.checker.check_model(onnx_model)
with open(graph_txt_path, 'w') as f:
    f.write(str(onnx_model.graph))

# Define the function to perform human detection on a video
def detect_human(video_path, output_dir):
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, video_name)
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    colors = [(0, 255, 0), (0, 0, 255)]  # Green, Red
    color_index = 0
    
    # Dictionary to store color assigned to each detected person
    person_colors = {}
    
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

        # Draw bounding boxes around detected humans with different colors
        for box in human_boxes:
            box = box.detach().cpu().numpy().astype(int)
            # Check if the person is already assigned a color
            if tuple(box) in person_colors:
                color = person_colors[tuple(box)]
            else:
                color = colors[color_index % len(colors)]  # Get color from list cyclically
                person_colors[tuple(box)] = color  # Store color for this person
                color_index += 1  # Increment color index for the next person
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Define the paths for input and output directories
# input_dir = r'E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\dataset\train_3'
# output_dir = r'E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\dataset\output_person_tracking'

input_dir = r'E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\Ch4'
output_dir = r'E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\ch4_output_person_detection'

# Iterate through all video files in the input directory
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.mp4'):
            video_path = os.path.join(root, file)
            detect_human(video_path, output_dir)
