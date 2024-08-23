# import os
# import cv2

# def extract_frames(video_path, output_dir):
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)
    
#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Initialize frame count
#     # frame_count = 107
#     frame_count = 195
    
#     # Read until video is completed
#     while cap.isOpened():
#         # Read a single frame from the video
#         ret, frame = cap.read()
        
#         if ret:
#             # Construct the output file path
#             output_file = os.path.join(output_dir, f"hugging_{frame_count}.jpg")
            
#             # Save the frame as an image
#             cv2.imwrite(output_file, frame)
            
#             # Increment frame count
#             frame_count += 1
#         else:
#             break
    
#     # Release the video capture object
#     cap.release()
    
#     print(f"Extracted {frame_count} frames from {video_path}")

# # Path to the directory containing the videos
# videos_dir = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\output_person_tracking"

# # Path to the directory where extracted frames will be saved
# output_dir = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\crc_output_person_frames"

# # Iterate through all video files in the input directory
# for root, dirs, files in os.walk(videos_dir):
#     for file in files:
#         if file.endswith('.mp4') or file.endswith('.avi'):
#             video_path = os.path.join(root, file)
#             extract_frames(video_path, output_dir)


# extract all to frame

# import os
# import cv2

# def extract_frames(video_path, output_dir):
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)
    
#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Initialize frame count
#     frame_count = 0
    
#     # Read until video is completed
#     while cap.isOpened():
#         # Read a single frame from the video
#         ret, frame = cap.read()
        
#         if ret:
#             # Construct the output file path
#             output_file = os.path.join(output_dir, f"none_interaction_{frame_count}.jpg")
            
#             # Save the frame as an image
#             cv2.imwrite(output_file, frame)
            
#             # Increment frame count
#             frame_count += 1
#         else:
#             break
    
#     # Release the video capture object
#     cap.release()
    
#     print(f"Extracted {frame_count} frames from {video_path}")

# # Path to the directory containing the videos
# videos_dir = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\crc_output_person_tracking\none_interaction"

# # Path to the directory where extracted frames will be saved
# output_dir = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\none_interaction_output_frames"

# # Iterate through all video files in the input directory
# for file in os.listdir(videos_dir):
#     if file.endswith('.mp4') or file.endswith('.avi'):
#         video_path = os.path.join(videos_dir, file)
#         output_subdir = os.path.join(output_dir, os.path.splitext(file)[0])
#         extract_frames(video_path, output_subdir)


# sort number of file

import os

# Path to the folder containing the images
folder_path = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\all_channels_output_frame\pushing\c4\pushing_c4_20"

# Get a list of all image files in the folder
image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]

# Starting index for renaming
start_index = 673
# Iterate over the image files and rename them
for i, file in enumerate(image_files):
    # Construct the new file name
    new_file_name = f"{start_index + i}.jpg"
    
    # Construct the full paths for the old and new files
    old_file_path = os.path.join(folder_path, file)
    new_file_path = os.path.join(folder_path, new_file_name)
    
    # Rename the file
    os.rename(old_file_path, new_file_path)

    print(f"Renamed {file} to {new_file_name}")

print("All files renamed successfully.")


