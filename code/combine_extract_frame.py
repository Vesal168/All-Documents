# import os
# import cv2

# def extract_frames(video_path, output_dir):

#     cap = cv2.VideoCapture(video_path)
#     os.makedirs(output_dir, exist_ok=True)
   
#     frame_count = 0
    
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         if ret:
#             output_file = os.path.join(output_dir, f"hugging_{frame_count}.jpg")
            
#             cv2.imwrite(output_file, frame)
#             frame_count += 1

#         else:
#             break
    
#     cap.release()

#     print(f"Extracted {frame_count} frames from {video_path}")

# videos_dir = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\Ch4\hugging"
# output_dir = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\ch4-output-frames"

# for file in os.listdir(videos_dir):
#     if file.endswith('.mp4') or file.endswith('.avi'):
#         video_path = os.path.join(videos_dir, file)
#         output_subdir = os.path.join(output_dir, os.path.splitext(file)[0])
#         extract_frames(video_path, output_subdir)


#no limit frame


# import os
# import cv2

# def extract_frames(video_path, output_dir):
#     cap = cv2.VideoCapture(video_path)
#     os.makedirs(output_dir, exist_ok=True)
#     frame_count = 0
    
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         if ret:
#             output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_{frame_count}.jpg")
#             cv2.imwrite(output_file, frame)
#             frame_count += 1
#         else:
#             break
    
#     cap.release()
#     print(f"Extracted {frame_count} frames from {video_path}")

# videos_dir = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\ch4"
# output_dir = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\ch4-output-frames\pushing"

# for class_name in os.listdir(videos_dir):
#     class_folder = os.path.join(videos_dir, class_name)
#     if os.path.isdir(class_folder):
#         for video_filename in os.listdir(class_folder):
#             if video_filename.endswith('.mp4'):
#                 video_path = os.path.join(class_folder, video_filename)
#                 extract_frames(video_path, output_dir)


#limit frame 60

# import os
# import cv2

# def extract_frames(video_path, output_dir, max_frames=61):
#     cap = cv2.VideoCapture(video_path)
#     os.makedirs(output_dir, exist_ok=True)
#     frame_count = 0
    
#     while cap.isOpened() and frame_count < max_frames:
#         ret, frame = cap.read()
        
#         if ret:
#             output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_{frame_count}.jpg")
#             cv2.imwrite(output_file, frame)
#             frame_count += 1
#         else:
#             break
    
#     cap.release()
#     print(f"Extracted {frame_count} frames from {video_path}")

# videos_dir = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\trim_video"
# output_dir = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\ch4-output-frames-170\hugging"

# for class_name in os.listdir(videos_dir):
#     class_folder = os.path.join(videos_dir, class_name)
#     if os.path.isdir(class_folder):
#         for video_filename in os.listdir(class_folder):
#             if video_filename.endswith('.mp4'):
#                 video_path = os.path.join(class_folder, video_filename)
#                 extract_frames(video_path, output_dir)


#limit 30

import os
import cv2

def extract_frames(video_path, output_dir, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        
        if ret:
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_{frame_count}.jpg")
            cv2.imwrite(output_file, frame)
            frame_count += 1
        else:
            break
    
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

videos_dir = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\trim_video"
output_dir = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\ch4-output-frame-"

for class_name in os.listdir(videos_dir):
    class_folder = os.path.join(videos_dir, class_name)
    if os.path.isdir(class_folder):
        for video_filename in os.listdir(class_folder):
            if video_filename.endswith('.mp4'):
                video_path = os.path.join(class_folder, video_filename)
                extract_frames(video_path, output_dir)

