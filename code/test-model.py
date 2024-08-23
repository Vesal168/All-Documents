import cv2
import numpy as np
import joblib
import pandas as pd

loaded_model = joblib.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\src\model\rf_model.pkl")
test_feature_vectors = pd.read_csv(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\asset\test_video\pointing41_test5_feature_vector.csv", header=None).to_numpy()
test_labels = pd.read_csv(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\asset\test_video\pointing41_test5_feature_vector_label.csv")['label'].values

label_to_index = {label: idx for idx, label in enumerate(np.unique(test_labels))}

# video_path = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\asset\child_mom.mp4"
video_path = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\Ch4\pointing\pointing_41.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 255, 0) 
thickness = 2
text_position = (10, 30) 

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = r'E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\asset\test_video\pointing41_test5_output.avi'
output_fps = 30.0 
output_frame_size = (width, height)

out = cv2.VideoWriter(output_video_path, fourcc, output_fps, output_frame_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)

    predicted_label = loaded_model.predict(test_feature_vectors)
    predicted_class = np.array([label_to_index.get(label, -1) for label in predicted_label])
    predicted_label = test_labels[predicted_class]

    cv2.putText(frame, str(predicted_label[0]), text_position, font, font_scale, font_color, thickness)
    cv2.imshow("Video", frame)

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

out.release()
cv2.destroyAllWindows()