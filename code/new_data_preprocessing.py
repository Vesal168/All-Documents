
# split data into 80, 10, 10

# import os
# import numpy as np
# from sklearn.model_selection import train_test_split

# # Define the path to your main folder (keypoint)
# # main_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\keypoints_extracted_clean"
# main_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\keypoints_extracted_clean_after_crop"

# # Initialize lists to store data and labels
# data = []
# labels = []

# # Iterate through each subfolder

# #  define label: ['hugging': 0, 'kicking': 1, 'pushing': 2, 'punching': 3, 'pointing': 4, 'none-interaction': 5]
# #dir_labels = {""}


# for subfolder in os.listdir(main_folder):  # class
#     subfolder_path = os.path.join(main_folder, subfolder)
#     for file in os.listdir(subfolder_path):
#         file_path = os.path.join(subfolder_path, file)
#         data_file = np.load(file_path)
#         data.append(data_file)
#         if subfolder == "hugging":  
#             labels.append(0)

#         elif subfolder == "kicking":  
#             labels.append(1)

#         elif subfolder == "pushing":  
#             labels.append(2)

#         elif subfolder == "punching":  
#             labels.append(3)

#         elif subfolder == "pointing":  
#             labels.append(4)

#         elif subfolder == "none-interaction":  
#             labels.append(5)

#     ''' if os.path.isdir(subfolder_path):  # wrong
#         # Load numpy files from each subfolder
#         for file in os.listdir(subfolder_path):
#             if file.endswith(".npy"):
#                 file_path = os.path.join(subfolder_path, file)
#                 keypoints = np.load(file_path)  # Load numpy data
#                 data.append(keypoints)
#                 labels.append(subfolder)  # Use subfolder name as label'''

# # Convert lists to numpy arrays
# data = np.array(data)
# labels = np.array(labels)

# # Split data into train, validation, and test sets
# X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.4, random_state=42)
# X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# # Now you have X_train, y_train (80% of data), X_valid, y_valid (10% of data), and X_test, y_test (10% of data)
# print("Train data shape:", X_train.shape)
# print("Y train shape:", y_train.shape)
# print("Validation data shape:", X_valid.shape)
# print("Y Valid shape:", y_valid.shape)
# print("Test data shape:", X_test.shape)
# print("Y test shape:", y_test.shape)


# output_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video"
# os.makedirs(output_folder, exist_ok=True)
# np.save(os.path.join(output_folder, "X_train.npy"), X_train)
# np.save(os.path.join(output_folder, "y_train.npy"), y_train)
# # np.save(os.path.join(output_folder, "X_valid.npy"), X_valid)
# # np.save(os.path.join(output_folder, "y_valid.npy"), y_valid)
# np.save(os.path.join(output_folder, "X_test.npy"), X_test)
# np.save(os.path.join(output_folder, "y_test.npy"), y_test)

# print("Data saved successfully in the 'split_data' folder!")



#split data into train and test 70-30 with after cropped video 110 video

# import os
# import numpy as np
# from sklearn.model_selection import train_test_split

# # Define the path to your main folder (keypoint)
# main_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\keypoints_extracted_clean_after_crop"

# # Initialize lists to store data and labels
# data = []
# labels = []

# # Iterate through each subfolder
# for subfolder in os.listdir(main_folder):
#     subfolder_path = os.path.join(main_folder, subfolder)
#     for file in os.listdir(subfolder_path):
#         file_path = os.path.join(subfolder_path, file)
#         data_file = np.load(file_path)
#         data.append(data_file)
#         if subfolder == "hugging":
#             labels.append(0)
#         elif subfolder == "kicking":
#             labels.append(1)
#         elif subfolder == "pushing":
#             labels.append(2)
#         elif subfolder == "pointing":
#             labels.append(3)
#         elif subfolder == "none-interaction":
#             labels.append(4)

# # Convert lists to numpy arrays
# data = np.array(data)
# labels = np.array(labels)

# # Split data into train and test sets (70% train, 30% test)
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# # Now you have X_train, y_train (70% of data) and X_test, y_test (30% of data)
# print("Train data shape:", X_train.shape)
# print("Y train shape:", y_train.shape)
# print("Test data shape:", X_test.shape)
# print("Y test shape:", y_test.shape)

# # Define the output folder
# output_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes"

# # Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Save the train and test data
# np.save(os.path.join(output_folder, "X_train.npy"), X_train)
# np.save(os.path.join(output_folder, "y_train.npy"), y_train)
# np.save(os.path.join(output_folder, "X_test.npy"), X_test)
# np.save(os.path.join(output_folder, "y_test.npy"), y_test)

# print("Data saved successfully in the 'split_data' folder!")


#split data into train and test 70-30 with after cropped video (200 video each classes)

# import os
# import numpy as np
# from sklearn.model_selection import train_test_split

# # Define the path to your main folder (keypoint)
# main_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\keypoint-extracted-clean-200perclass-clean"

# # Initialize lists to store data and labels
# data = []
# labels = []

# # Iterate through each subfolder
# for subfolder in os.listdir(main_folder):
#     subfolder_path = os.path.join(main_folder, subfolder)
#     for file in os.listdir(subfolder_path):
#         file_path = os.path.join(subfolder_path, file)
#         data_file = np.load(file_path)
#         data.append(data_file)
#         if subfolder == "hugging":
#             labels.append(0)
#         elif subfolder == "kicking":
#             labels.append(1)
#         elif subfolder == "pushing":
#             labels.append(2)
#         elif subfolder == "pointing":
#             labels.append(3)
#         elif subfolder == "none-interaction":
#             labels.append(4)

# # Convert lists to numpy arrays
# data = np.array(data)
# labels = np.array(labels)

# # Split data into train and test sets (70% train, 30% test)
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# # Now you have X_train, y_train (70% of data) and X_test, y_test (30% of data)
# print("Train data shape:", X_train.shape)
# print("Y train shape:", y_train.shape)
# print("Test data shape:", X_test.shape)
# print("Y test shape:", y_test.shape)

# # Define the output folder
# output_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-70-30-witth-public-200perclass"

# # Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Save the train and test data
# np.save(os.path.join(output_folder, "X_train.npy"), X_train)
# np.save(os.path.join(output_folder, "y_train.npy"), y_train)
# np.save(os.path.join(output_folder, "X_test.npy"), X_test)
# np.save(os.path.join(output_folder, "y_test.npy"), y_test)

# print("Data saved successfully in the 'split_data' folder!")


#split data into 60 and 40 for old data 110 video croped

import os
import numpy as np
from sklearn.model_selection import train_test_split

# Define the path to your main folder (keypoint)
main_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\keypoints_extracted_clean_after_crop"

# Initialize lists to store data and labels
data = []
labels = []

# Iterate through each subfolder
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    for file in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, file)
        data_file = np.load(file_path)
        data.append(data_file)
        if subfolder == "hugging":
            labels.append(0)
        elif subfolder == "pushing":
            labels.append(1)
        elif subfolder == "pointing":
            labels.append(2)
        elif subfolder == "kicking":
            labels.append(3)
        elif subfolder == "none-interaction":
            labels.append(4)

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Now you have X_train, y_train (80% of data) and X_test, y_test (20% of data)
print("Train data shape:", X_train.shape)
print("Y train shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Y test shape:", y_test.shape)

# Define the output folder
output_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save the train and test data
np.save(os.path.join(output_folder, "X_train.npy"), X_train)
np.save(os.path.join(output_folder, "y_train.npy"), y_train)
np.save(os.path.join(output_folder, "X_test.npy"), X_test)
np.save(os.path.join(output_folder, "y_test.npy"), y_test)

print("Data saved successfully in the 'split_data' folder!")








