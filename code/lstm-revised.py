
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Flatten, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

total_sample = 440
total_frames = 51
keypoints = 50
CLASSES_LIST = ['hugging', 'kicking' ,'pushing', 'pointing', 'none-interaction']

X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_train.npy")
X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_test.npy")

x_train_reshaped = np.reshape(X_train, (total_sample, total_frames, keypoints * 2))
x_test_reshaped = np.reshape(X_test, (110, total_frames, keypoints * 2))

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(total_frames, keypoints * 2)),
    # Dropout(0.5),

    LSTM(128, return_sequences=True),
    # Dropout(0.5),

    LSTM(128, return_sequences=True),
    # Dropout(0.5),

    LSTM(128, return_sequences=True),
    # Dropout(0.5),

    LSTM(128),
    Dropout(0.3),

    Dense(128, activation='relu'),

    Dense(len(CLASSES_LIST), activation='softmax')
])


model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(r'E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\src\model\lstm_best_model.keras', monitor='val_loss', mode='min', save_best_only=True)

# Train the model
history = model.fit(x_train_reshaped, y_train, epochs=300, batch_size=2, shuffle=True, validation_data=(x_test_reshaped, y_test), verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test_reshaped, y_test, verbose = 1)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# model.save(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\src\model\lstm-v3.h5")

# Plot confusion matrix
import time
start = time.time()
y_pred = model.predict(x_test_reshaped)
stop = time.time()
predict_duration = stop-start

y_pred_classes = np.argmax(y_pred, axis=1)

conf_matrix = confusion_matrix(y_test, y_pred_classes)
conf_matrix = conf_matrix / conf_matrix.astype(float).sum(axis=1) * 100

class_report = classification_report(y_test, y_pred_classes, target_names=CLASSES_LIST)
print("Classification Report:")
print("===========================================================")
print(class_report)
print("===========================================================")

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

