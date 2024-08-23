import pandas as pd
import numpy as np

def compute_angle(p1, p2, p3):
    angle_radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    return np.degrees(angle_radians) % 360

def compute_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def extract_features_from_keypoints(keypoints_df):
    feature_vectors = []
    labels = []
    for _, group in keypoints_df.groupby('label'):
        label = group.iloc[0]['label']
        labels.append(label)
        feature_vector = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                point1 = (group.iloc[i]['x'], group.iloc[i]['y'])
                point2 = (group.iloc[j]['x'], group.iloc[j]['y'])
                angle = compute_angle(point1, (0, 0), point2)
                distance = compute_distance(point1, point2)
                feature_vector.extend([angle, distance])

        while len(feature_vector) < 600:
            feature_vector.append(0)
        feature_vectors.append(feature_vector[:600])
    return feature_vectors, labels

keypoints_df = pd.read_csv(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\asset\test_video\test2_keypoints.csv")

feature_vectors, labels = extract_features_from_keypoints(keypoints_df)

feature_df = pd.DataFrame(feature_vectors)
feature_df.to_csv(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\asset\test_video\test2_feature_vector.csv", index=False, header=False)

labels_df = pd.DataFrame({'label': labels})
labels_df.to_csv(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\asset\test_video\test2_feature_vector_labels.csv", index=False)
