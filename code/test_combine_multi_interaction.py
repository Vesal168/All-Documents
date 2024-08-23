
import cv2
import numpy as np
import joblib
import pandas as pd


# loaded_model = joblib.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\src\model\svm_model.pkl")
loaded_model = joblib.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\src\model\rf_model.pkl")
test_feature_vectors = pd.read_csv(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\asset\mix_interaction_feature_vectors.csv", header=None).to_numpy()
label_data = pd.read_csv(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\asset\manual_label.csv")

def parse_timestamp(timestamp_str):

    parts = timestamp_str.split(':')
    minutes = int(parts[0])
    seconds = int(parts[1])
    
    total_seconds = minutes * 60 + seconds
    return total_seconds

timestamps = label_data["Timestamp"].apply(parse_timestamp).values
interactions = label_data["Interaction"].values

timestamps = np.array(timestamps, dtype=np.float64)

predicted_interactions = []
video_path = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\asset\mix_action.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    closest_timestamp_idx = np.argmin(np.abs(timestamps - current_timestamp))

    test_feature_vector = test_feature_vectors[closest_timestamp_idx]
    predicted_interaction = loaded_model.predict(test_feature_vector.reshape(1, -1))

    cv2.putText(frame, str(predicted_interaction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Video", frame)

    if cv2.waitKey(50) & 0xFF == ord("q"):
        break
    predicted_interactions.append(predicted_interaction)

cap.release()
cv2.destroyAllWindows()

print(predicted_interactions)




