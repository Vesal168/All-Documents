
# import pandas as pd
# import numpy as np
# from math import atan2, degrees

# # Function to compute angle between three points
# def compute_angle(p1, p2, p3):
#     angle_radians = atan2(p3[1] - p2[1], p3[0] - p2[0]) - atan2(p1[1] - p2[1], p1[0] - p2[0])
#     return np.degrees(angle_radians) % 360

# # Function to compute distance between two points
# def compute_distance(p1, p2):
#     return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# # Read keypoints from CSV
# keypoints_df = pd.read_csv(r"E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\dataset\txt_keypoint\hugging_keypoints.csv") 

# # Compute features
# max_keypoints = 25 
# feature_vectors = []
# for image_file, group in keypoints_df.groupby('Image File'):
#     feature_vector = []
#     for i in range(min(len(group), max_keypoints)):
#         for j in range(i+1, min(len(group), max_keypoints)):
#             point1 = (group.iloc[i]['X'], group.iloc[i]['Y'])
#             point2 = (group.iloc[j]['X'], group.iloc[j]['Y'])
#             angle = compute_angle(point1, (0,0), point2) 
#             distance = compute_distance(point1, point2)
#             feature_vector.extend([angle, distance])
#     # Pad feature vector with zeros if less than max_keypoints keypoints detected
#     if len(feature_vector) < int(max_keypoints * (max_keypoints - 1) / 2 * 2):
#         feature_vector.extend([0] * (int(max_keypoints * (max_keypoints - 1) / 2 * 2) - len(feature_vector)))
#     feature_vectors.append(feature_vector)

# # Convert feature_vectors to numpy array
# feature_vectors = np.array(feature_vectors)
# # print(feature_vectors)

# print("Shape of test_feature_vectors:", feature_vectors.shape)






# only test video

# import pandas as pd
# import numpy as np
# from math import atan2, degrees

# # Function to compute angle between three points
# def compute_angle(p1, p2, p3):
#     angle_radians = atan2(p3[1] - p2[1], p3[0] - p2[0]) - atan2(p1[1] - p2[1], p1[0] - p2[0])
#     return np.degrees(angle_radians) % 360

# # Function to compute distance between two points
# def compute_distance(p1, p2):
#     return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# # Read keypoints from CSV
# keypoints_df = pd.read_csv(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\asset\mix_interaction_keypoints.csv") 

# # Compute features
# max_keypoints = 25 
# feature_vectors = []
# for frame_number, group in keypoints_df.groupby('frame_number'):
#     feature_vector = []
#     for i in range(min(len(group), max_keypoints)):
#         for j in range(i+1, min(len(group), max_keypoints)):
#             point1 = (group.iloc[i]['x'], group.iloc[i]['y'])
#             point2 = (group.iloc[j]['x'], group.iloc[j]['y'])
#             angle = compute_angle(point1, (0,0), point2) 
#             distance = compute_distance(point1, point2)
#             feature_vector.extend([angle, distance])
#     # Pad feature vector with zeros if less than max_keypoints keypoints detected
#     if len(feature_vector) < int(max_keypoints * (max_keypoints - 1) / 2 * 2):
#         feature_vector.extend([0] * (int(max_keypoints * (max_keypoints - 1) / 2 * 2) - len(feature_vector)))
#     feature_vectors.append(feature_vector)

# # Convert feature_vectors to numpy array
# feature_vectors = np.array(feature_vectors)
# print(feature_vectors)



# only test video with new csv


import pandas as pd
import numpy as np

def compute_angle(p1, p2, p3):
    angle_radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    return np.degrees(angle_radians) % 360

def compute_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def extract_features_from_keypoints(keypoints_df):
    feature_vectors = []
    for _, group in keypoints_df.groupby('label'):
        feature_vector = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                point1 = (group.iloc[i]['x'], group.iloc[i]['y'])
                point2 = (group.iloc[j]['x'], group.iloc[j]['y'])
                angle = compute_angle(point1, (0,0), point2)
                distance = compute_distance(point1, point2)
                feature_vector.extend([angle, distance])
        feature_vectors.append(feature_vector)
    return np.array(feature_vectors)

keypoints_df = pd.read_csv(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\asset\test_keypoints_1111.csv")

feature_vectors = extract_features_from_keypoints(keypoints_df)
print(feature_vectors)


#test with lastest video


















