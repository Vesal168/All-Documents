import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

total_sample = 352
total_frames = 51
keypoints = 50
CLASSES_LIST = ['hugging', 'pushing', 'pointing', 'none-interaction']

# Load your data
X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-80-20-data-after-cropped-110video-4-classes\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-80-20-data-after-cropped-110video-4-classes\y_train.npy")
X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-80-20-data-after-cropped-110video-4-classes\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-80-20-data-after-cropped-110video-4-classes\y_test.npy")

# Convert input data to float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Reshape your data to match the model's input shape
x_train_reshaped = np.reshape(X_train, (total_sample, total_frames, keypoints * 2)).astype(np.float32)
x_test_reshaped = np.reshape(X_test, (88, total_frames, keypoints * 2)).astype(np.float32)

model = Sequential([
    GRU(128, return_sequences=True, input_shape=(total_frames, keypoints * 2)),
    Dropout(0.5),
    GRU(64, return_sequences=True),
    Dropout(0.5),
    GRU(64),
    Dropout(0.3),
    Dense(len(CLASSES_LIST), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, decay=1e-6), metrics=['accuracy'])

# Print model summary
model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(r'E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\src\model\best_model.keras', monitor='val_loss', mode='min', save_best_only=True)

# Train the model
history = model.fit(x_train_reshaped, y_train, epochs=800, batch_size=6, shuffle=True, validation_data=(x_test_reshaped, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test_reshaped, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

model.save(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\src\model\gru-v1.h5")

# Plot confusion matrix
y_pred = model.predict(x_test_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)

conf_matrix = confusion_matrix(y_test, y_pred_classes)
conf_matrix = conf_matrix / conf_matrix.astype(float).sum(axis=1) * 100

class_report = classification_report(y_test, y_pred_classes, target_names=CLASSES_LIST)
print("Classification Report:")
print("===============================================================")
print(class_report)
print("===============================================================")

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
