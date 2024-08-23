
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization, 
    GRU, Dropout, Dense
)
from keras.models import Sequential
import time

tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CLASSES_LIST = ['hugging', 'pushing', 'pointing', 'kicking', 'none-interaction']

# Load data
X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_train.npy")
X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_test.npy")

# Reshape data for 1D CNN
x_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], -1))
x_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], -1))

print("x_train_reshaped: ", x_train_reshaped.shape)

input_shape = (51, 100)  # 51 time steps, 100 features

# Define the 1D CNN-GRU model
model = Sequential([ 
    Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
    MaxPooling1D(pool_size=2),
    BatchNormalization(),
    Dropout(0.3),

    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    BatchNormalization(),
    Dropout(0.3),

    GRU(100, return_sequences=True, activation='relu'),
    Dropout(0.3),

    GRU(100, return_sequences=True, activation='relu'),
    Dropout(0.3),

    GRU(200, return_sequences=False, activation='relu'),
    Dropout(0.3),

    Dense(100, activation='relu'),
    Dropout(0.3),

    Dense(100, activation='relu'),
    Dropout(0.3),

    Dense(100, activation='relu'),
    Dropout(0.3),

    Dense(len(CLASSES_LIST), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

model.summary()

# Track the training time
start_time = time.time()

# Train the 1D CNN-GRU model
history = model.fit(x_train_reshaped, y_train, epochs=500, batch_size=4, shuffle=True, validation_data=(x_test_reshaped, y_test))

end_time = time.time()
training_time = end_time - start_time
print(f'Total Training Time: {training_time:.2f} seconds')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test_reshaped, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Predict and evaluate
y_pred = model.predict(x_test_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred_classes)
conf_matrix_normalized = conf_matrix.astype(float) / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

class_report = classification_report(y_test, y_pred_classes, target_names=CLASSES_LIST)
print("Classification Report:")
print("===========================================================")
print(class_report)
print("===========================================================")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
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
