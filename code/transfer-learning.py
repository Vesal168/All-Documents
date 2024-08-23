import os
import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define constants
total_sample = 462
total_frames = 51
keypoints = 50
CLASSES_LIST = ['hugging', 'kicking', 'pushing', 'punching', 'pointing', 'none-interaction']

# Load your data
X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data-clean\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data-clean\y_train.npy")
X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data-clean\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data-clean\y_test.npy")

# Convert input data to float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Reshape your data to match the model's input shape
# x_train_reshaped = np.reshape(X_train, (total_sample, total_frames, keypoints, 2, 1)).astype(np.float32)
# x_test_reshaped = np.reshape(X_test, (198, total_frames, keypoints, 2, 1)).astype(np.float32)
# x_train_reshaped = np.reshape(X_train, (462, 51, 50, 2)).astype(np.float32)
# x_test_reshaped = np.reshape(X_test, (198, 51, 50, 2)).astype(np.float32)

x_train_reshaped = np.reshape(X_train, (462, 51, 50, 33, 3)).astype(np.float32)

print(x_train_reshaped.shape)

# # Define label encoding function
# def encode_labels(labels, classes_list):
#     label_dict = {label: i for i, label in enumerate(classes_list)}
#     encoded_labels = [label_dict[label] for label in labels]
#     return np.array(encoded_labels)

# # Encode labels
# y_train_encoded = encode_labels(y_train, CLASSES_LIST)
# y_test_encoded = encode_labels(y_test, CLASSES_LIST)

# # Load pre-trained ResNet50 model without the top layer
# resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(None, None, 2))

# # Freeze the layers of the ResNet50 model
# for layer in resnet.layers:
#     layer.trainable = False

# # Create a new model on top of the ResNet50 base model
# model = Sequential([
#     resnet,
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dropout(0.3),
#     Dense(len(CLASSES_LIST), activation='softmax')
# ])

# # Compile the model
# model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0005, decay=1e-6), metrics=['accuracy'])

# # Print model summary
# model.summary()

# # Train the model
# history = model.fit(x_train_reshaped, y_train_encoded, epochs=200, batch_size=8, shuffle=True, validation_data=(x_test_reshaped, y_test_encoded))

# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(x_test_reshaped, y_test_encoded)
# print(f'Test Loss: {test_loss}')
# print(f'Test Accuracy: {test_accuracy}')

# # Predict on test data
# y_pred = model.predict(x_test_reshaped)
# y_pred_classes = np.argmax(y_pred, axis=1)

# # Plot confusion matrix
# conf_matrix = confusion_matrix(y_test_encoded, y_pred_classes)
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()

# # Plot training and validation loss
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()
