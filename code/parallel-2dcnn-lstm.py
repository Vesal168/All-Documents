import os
import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, LSTM, Conv2D, MaxPooling2D, BatchNormalization, Reshape, Input, concatenate, GlobalAveragePooling2D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# GPU configuration
tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define class labels
CLASSES_LIST = ['hugging', 'pushing', 'pointing', 'kicking', 'none-interaction']

# Load data
X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_train.npy")
X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_test.npy")

# Reshape data
x_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 50, 2))
x_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 50, 2))

print("x_train_reshaped: ", x_train_reshaped.shape)

input_shape = (51, 50, 2)

# Define the CNN model
inputs_cnn = Input(shape=input_shape)
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs_cnn)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

cnn_output = GlobalAveragePooling2D()(x)

cnn_model = Model(inputs_cnn, cnn_output)

# Define the LSTM model
inputs_lstm = Input(shape=input_shape)
y = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs_lstm)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = BatchNormalization()(y)
y = Dropout(0.3)(y)

y = Conv2D(64, kernel_size=(3, 3), activation='relu')(y)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = BatchNormalization()(y)
y = Dropout(0.3)(y)

y = Reshape((11, 704))(y)  # Reshape for LSTM
y = LSTM(100, return_sequences=True, activation='relu')(y)
y = Dropout(0.3)(y)
y = LSTM(200, return_sequences=False, activation='relu')(y)
y = Dropout(0.3)(y)

lstm_model = Model(inputs_lstm, y)

# Concatenate features from both models
combined_features = concatenate([cnn_model.output, lstm_model.output])

# Add fully connected layers with regularization
z = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(combined_features)
z = Dropout(0.5)(z)
z = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(z)
z = Dropout(0.5)(z)
z = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(z)
z = Dropout(0.5)(z)
outputs = Dense(len(CLASSES_LIST), activation='softmax')(z)

# Define the combined model
combined_model = Model(inputs=[inputs_cnn, inputs_lstm], outputs=outputs)

combined_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

combined_model.summary()

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# Train the combined model
history = combined_model.fit([x_train_reshaped, x_train_reshaped], y_train, epochs=300, batch_size=4, shuffle=True, validation_data=([x_test_reshaped, x_test_reshaped], y_test))

# Evaluate the model
test_loss, test_accuracy = combined_model.evaluate([x_test_reshaped, x_test_reshaped], y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Predict and evaluate
y_pred = combined_model.predict([x_test_reshaped, x_test_reshaped])
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

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
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
