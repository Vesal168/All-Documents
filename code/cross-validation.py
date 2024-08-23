import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Reshape, GRU, Dropout, Dense
from keras.models import Sequential, Model
from sklearn.model_selection import KFold
import time

# GPU Configuration
tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Class labels
CLASSES_LIST = ['hugging', 'pushing', 'pointing', 'kicking', 'none-interaction']

# Load data
X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_train.npy")
X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_test.npy")

# Reshape data
x_train_reshaped = np.reshape(X_train, (X_train.shape[0], 51, 50, 2))
x_test_reshaped = np.reshape(X_test, (X_test.shape[0], 51, 50, 2))

print("x_train_reshaped: ", x_train_reshaped.shape)

input_shape = (51, 50, 2)

# Function to create a new instance of the model
def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(0.3),

        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(0.3),

        Reshape((11, 704)), 
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

        Dense(len(CLASSES_LIST), activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model

# Cross-validation setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_no = 1
epochs_per_fold = 200
batch_size = 4

# Track training time
start_time = time.time()

# Lists to store results
fold_accuracies = []

# Cross-validation loop
for train_index, val_index in kf.split(x_train_reshaped):
    print(f"Training on fold {fold_no}...")
    
    # Split data
    X_train_fold, X_val_fold = x_train_reshaped[train_index], x_train_reshaped[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Create a new model instance
    model = create_model()
    
    # Train the model
    history = model.fit(X_train_fold, y_train_fold, epochs=epochs_per_fold, batch_size=batch_size, shuffle=True, validation_data=(X_val_fold, y_val_fold), verbose=1)
    
    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold)
    fold_accuracies.append(val_accuracy)
    print(f"Fold {fold_no} - Validation Accuracy: {val_accuracy * 100:.2f}%")
    
    fold_no += 1

end_time = time.time()
training_time = end_time - start_time
print(f'Total Training Time: {training_time:.2f} seconds')

# Calculate average accuracy across all folds
average_accuracy = np.mean(fold_accuracies)
print("\nCross-Validation Results:")
print("===========================================================")
for i, acc in enumerate(fold_accuracies, 1):
    print(f"Fold {i}: {acc * 100:.2f}%")
print("===========================================================")
print(f"Average Cross-Validation Accuracy: {average_accuracy * 100:.2f}%")

# Train final model on the entire training data for the specified number of epochs
final_model = create_model()
final_model.fit(x_train_reshaped, y_train, epochs=200, batch_size=batch_size, shuffle=True, validation_data=(x_test_reshaped, y_test))

# Evaluate the final model on the test set
test_loss, test_accuracy = final_model.evaluate(x_test_reshaped, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Predict and evaluate on the test set
y_pred = final_model.predict(x_test_reshaped)
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
