
# import os
# import numpy as np
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
# os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES']= '2'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# total_sample = 385
# total_frames = 51
# keypoints = 50
# CLASSES_LIST = ['hugging', 'kicking', 'pushing', 'pointing', 'none-interaction']

# # Load your data
# # X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data-interpolate\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data-interpolate\y_train.npy")
# # X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data-interpolate\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data-interpolate\y_test.npy")

# # X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_train.npy")
# # X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_test.npy")

# X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\y_train.npy")
# X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\y_test.npy")

# # Convert input data to float32
# X_train = X_train.astype(np.float32)
# X_test = X_test.astype(np.float32)

# # Reshape your data to match the model's input shape
# # x_train_reshaped = np.reshape(X_train, (total_sample, total_frames, keypoints, 2, 1)).astype(np.float32)
# # x_test_reshaped = np.reshape(X_test, (198, total_frames, keypoints, 2, 1)).astype(np.float32)

# x_train_reshaped = np.reshape(X_train, (total_sample, total_frames, keypoints, 2, 1)).astype(np.float32)
# x_test_reshaped = np.reshape(X_test, (165, total_frames, keypoints ,2, 1)).astype(np.float32)

# # Define label encoding function
# # def encode_labels(labels, classes_list):
# #     label_dict = {label: i for i, label in enumerate(classes_list)}
# #     encoded_labels = [label_dict[label] for label in labels]
# #     return np.array(encoded_labels)

# # # Encode labels
# # y_train_encoded = encode_labels(y_train, CLASSES_LIST)
# # y_test_encoded = encode_labels(y_test, CLASSES_LIST)


# model = Sequential([
#     Conv3D(32, kernel_size=(3, 3, 3), input_shape=(total_frames, keypoints, 2, 1), activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
    
#     Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
    
#     Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
    
#     Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
    
#     Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
#     Dropout(0.3),

#     Flatten(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(len(CLASSES_LIST), activation='softmax')
# ])

# # Compile the model
# model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001, decay=1e-6), metrics=['accuracy'])

# # Print model summary
# model.summary()

# # Train the model
# # history = model.fit(x_train_reshaped, y_train_encoded, epochs=200, batch_size=8, shuffle=True, validation_data=(x_valid_reshaped, y_valid_encoded))
# history = model.fit(x_train_reshaped, y_train, epochs=100, batch_size=8, shuffle=True, validation_data=(x_test_reshaped, y_test))

# test_loss, test_accuracy = model.evaluate(x_test_reshaped, y_test)
# print(f'Test Loss: {test_loss}')
# print(f'Test Accuracy: {test_accuracy}')

# y_pred = model.predict(x_test_reshaped)
# y_pred_classes = np.argmax(y_pred, axis=1)

# conf_matrix = confusion_matrix(y_test, y_pred_classes)
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv3D, MaxPooling3D, BatchNormalization, Reshape
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from sklearn.model_selection import train_test_split

# GPU configuration
tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define class labels
CLASSES_LIST = ['hugging', 'kicking', 'pushing', 'pointing', 'none-interaction']

# Load data
X_train, y_train = np.load(r"Z:\Vesal\Human-Human-Interaction\dataset\split-data-80-20-5-110video\X_train.npy"), np.load(r"Z:\Vesal\Human-Human-Interaction\dataset\split-data-80-20-5-110video\y_train.npy")
X_test, y_test = np.load(r"Z:\Vesal\Human-Human-Interaction\dataset\split-data-80-20-5-110video\X_test.npy"), np.load(r"Z:\Vesal\Human-Human-Interaction\dataset\split-data-80-20-5-110video\y_test.npy")

# Reshape data
x_train_reshaped = np.reshape(X_train, (X_train.shape[0], 51, 50, 2, 1))
x_test_reshaped = np.reshape(X_test, (X_test.shape[0], 51, 50, 2, 1))

print("x_train_reshaped: ", x_train_reshaped.shape)

input_shape = (51, 50, 2, 1)

# Define the 3D CNN-LSTM model for feature extraction
model = Sequential([
    Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape),
    MaxPooling3D(pool_size=(2, 2, 2)),
    BatchNormalization(),
    Dropout(0.3),

    Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 2)),
    BatchNormalization(),
    Dropout(0.3),

    Reshape((51, -1)),  # Reshape for LSTM

    LSTM(100, return_sequences=True, activation='relu'),
    Dropout(0.3),

    LSTM(200, return_sequences=False, activation='relu'),
    Dropout(0.3),

    Dense(100, activation='relu'),
    Dropout(0.3),

    Dense(100, activation='relu'),
    Dropout(0.3),

    Dense(len(CLASSES_LIST), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

model.summary()

# Train the 3D CNN-LSTM model
history = model.fit(x_train_reshaped, y_train, epochs=100, batch_size=32, shuffle=True, validation_data=(x_test_reshaped, y_test), verbose=1)

# Extract features using the trained 3D CNN-LSTM model
feature_extractor = Sequential(model.layers[:-1])  # Remove the last softmax layer
train_features = feature_extractor.predict(x_train_reshaped)
test_features = feature_extractor.predict(x_test_reshaped)

# Split the features for training the Random Forest
X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(train_features, y_train, test_size=0.2, random_state=42)

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_rf, y_train_rf)

# Evaluate the Random Forest classifier
y_val_pred_rf = rf_classifier.predict(X_val_rf)
val_accuracy_rf = accuracy_score(y_val_rf, y_val_pred_rf)
print(f'Validation Accuracy (Random Forest): {val_accuracy_rf * 100:.2f}%')

y_test_pred_rf = rf_classifier.predict(test_features)
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
print(f'Test Accuracy (Random Forest): {test_accuracy_rf * 100:.2f}%')

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_test_pred_rf)
conf_matrix_normalized = conf_matrix.astype(float) / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

class_report = classification_report(y_test, y_test_pred_rf, target_names=CLASSES_LIST)

print("Classification Report:")
print("===========================================================")
print(class_report)
print("===========================================================")

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()



