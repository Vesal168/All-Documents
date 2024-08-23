# import os
# import numpy as np
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM, Conv2D, MaxPooling2D, BatchNormalization, Reshape
# import matplotlib.pyplot as plt
# from sklearn.svm import SVC
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # GPU configuration
# tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Define class labels
# CLASSES_LIST = ['hugging', 'kicking', 'pushing', 'pointing', 'none-interaction']

# # Load data
# X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_train.npy")
# X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_test.npy")

# # Reshape data
# x_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 50, 2))
# x_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 50, 2))

# print("x_train_reshaped: ", x_train_reshaped.shape)

# input_shape = (51, 50, 2)

# # Define the CNN-LSTM model for feature extraction
# model = Sequential([ 
#     Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),
#     Dropout(0.3),
#     Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),
#     Dropout(0.3),
#     Reshape((11, 704)),  # Reshape for LSTM
#     LSTM(100, return_sequences=True, activation='relu'),
#     Dropout(0.3),
#     LSTM(200, return_sequences=False, activation='relu'),
#     Dropout(0.3),
#     Dense(100, activation='relu'),
#     Dropout(0.3),
#     Dense(100, activation='relu'),
#     Dropout(0.3),
#     Dense(len(CLASSES_LIST), activation='softmax')
# ])

# model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# model.summary()

# # Train the CNN-LSTM model
# history = model.fit(x_train_reshaped, y_train, epochs=300, batch_size=4, shuffle=True, validation_data=(x_test_reshaped, y_test))

# # Extract features using the trained CNN-LSTM model
# feature_extractor = Sequential(model.layers[:-1])  # Remove the last softmax layer
# train_features = feature_extractor.predict(x_train_reshaped)
# test_features = feature_extractor.predict(x_test_reshaped)

# # Standardize the features
# scaler = StandardScaler()
# train_features = scaler.fit_transform(train_features)
# test_features = scaler.transform(test_features)

# # Split the features for training the SVM
# X_train_svm, X_val_svm, y_train_svm, y_val_svm = train_test_split(train_features, y_train, test_size=0.2, random_state=42)

# # Train the SVM classifier
# svm_classifier = SVC(kernel='linear', probability=True)
# svm_classifier.fit(X_train_svm, y_train_svm)

# # Evaluate the SVM classifier
# y_val_pred_svm = svm_classifier.predict(X_val_svm)
# val_accuracy_svm = accuracy_score(y_val_svm, y_val_pred_svm)
# print(f'Validation Accuracy (SVM): {val_accuracy_svm * 100:.2f}%')

# y_test_pred_svm = svm_classifier.predict(test_features)
# test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)
# print(f'Test Accuracy (SVM): {test_accuracy_svm * 100:.2f}%')

# # Confusion matrix and classification report
# conf_matrix = confusion_matrix(y_test, y_test_pred_svm)
# conf_matrix_normalized = conf_matrix.astype(float) / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# class_report = classification_report(y_test, y_test_pred_svm, target_names=CLASSES_LIST)

# print("Classification Report:")
# print("===========================================================")
# print(class_report)
# print("===========================================================")

# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix (Normalized)')
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


import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv2D, MaxPooling2D, BatchNormalization, Reshape
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

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

# Define the CNN-LSTM model for feature extraction
model = Sequential([ 
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.3),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.3),
    Reshape((11, 704)),  # Reshape for LSTM
    LSTM(100, return_sequences=True, activation='relu'),
    Dropout(0.3),
    LSTM(200, return_sequences=False, activation='relu'),
    Dropout(0.3),
    Dense(100, activation='relu'),
    Dropout(0.3),
    Dense(100, activation='relu', ),
    Dropout(0.3),
    Dense(len(CLASSES_LIST), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

model.summary()

# Train the CNN-LSTM model
history = model.fit(x_train_reshaped, y_train, epochs=100, batch_size=10, shuffle=True, validation_data=(x_test_reshaped, y_test))

# Extract features using the trained CNN-LSTM model
feature_extractor = Sequential(model.layers[:-1])  # Remove the last softmax layer
train_features = feature_extractor.predict(x_train_reshaped)
test_features = feature_extractor.predict(x_test_reshaped)

# Standardize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Split the features for training and validation
X_train_ml, X_val_ml, y_train_ml, y_val_ml = train_test_split(train_features, y_train, test_size=0.2, random_state=42)

# Function to train and evaluate a classifier
def train_and_evaluate_classifier(clf, X_train, y_train, X_val, y_val, X_test, y_test, classifier_name):
    clf.fit(X_train, y_train)
    
    y_val_pred = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Validation Accuracy ({classifier_name}): {val_accuracy * 100:.2f}%')

    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy ({classifier_name}): {test_accuracy * 100:.2f}%')

    conf_matrix = confusion_matrix(y_test, y_test_pred)
    conf_matrix_normalized = conf_matrix.astype(float) / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    class_report = classification_report(y_test, y_test_pred, target_names=CLASSES_LIST)

    print(f"Classification Report ({classifier_name}):")
    print("===========================================================")
    print(class_report)
    print("===========================================================")

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix (Normalized) - {classifier_name}')
    plt.show()

# Train and evaluate classifiers
classifiers = {
    'LightGBM': lgb.LGBMClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB()
}

for classifier_name, clf in classifiers.items():
    train_and_evaluate_classifier(clf, X_train_ml, y_train_ml, X_val_ml, y_val_ml, test_features, y_test, classifier_name)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
