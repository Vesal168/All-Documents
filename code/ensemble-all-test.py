
# import os
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# import seaborn as sns
# from keras.layers import (
#     Conv2D, MaxPooling2D, BatchNormalization, 
#     Dropout, Dense, Input, Add, Activation, GlobalAveragePooling2D
# )
# from keras.models import Model
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import PCA
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# from sklearn.naive_bayes import GaussianNB
# import time
# import joblib

# # GPU Configuration
# tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Define Class
# CLASSES_LIST = ['hugging', 'pushing', 'pointing', 'kicking', 'none-interaction']

# # Load data
# X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_train.npy")
# X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_test.npy")

# # Reshape data for input 
# x_train_reshaped = np.reshape(X_train, (X_train.shape[0], 51, 50, 2))
# x_test_reshaped = np.reshape(X_test, (X_test.shape[0], 51, 50, 2))
# print("x_train_reshaped: ", x_train_reshaped.shape)

# # Input shape format
# input_shape = (51, 50, 2)

# # Define the HI-ResNet block
# def resnet_block(inputs, filters, kernel_size, strides):
#     x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
#     x = BatchNormalization()(x)

#     shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(inputs)
#     shortcut = BatchNormalization()(shortcut)

#     x = Add()([x, shortcut])
#     x = Activation('relu')(x)

#     return x

# inputs = Input(shape=input_shape)

# # Create HI-ResNet model
# x = resnet_block(inputs, filters=32, kernel_size=(3, 3), strides=1)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.2)(x)

# x = resnet_block(x, filters=64, kernel_size=(3, 3), strides=1)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.2)(x)

# x = resnet_block(x, filters=128, kernel_size=(3, 3), strides=1)
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.2)(x)

# x = Dense(100, activation='relu')(x)
# x = Dropout(0.2)(x)

# x = Dense(100, activation='relu')(x)
# x = Dropout(0.2)(x)

# outputs = Dense(len(CLASSES_LIST), activation='softmax')(x)
# resnet_model = Model(inputs, outputs)

# resnet_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
# resnet_model.summary()

# # Track the start training time
# start_time = time.time()

# # Train the HI-ResNet model
# history = resnet_model.fit(x_train_reshaped, y_train, epochs=300, batch_size=4, shuffle=True, validation_data=(x_test_reshaped, y_test))

# # Track the end training time
# end_time = time.time()
# training_time = end_time - start_time
# print(f'Total Training Time: {training_time:.2f} seconds')

# # Evaluate the model
# test_loss, test_accuracy = resnet_model.evaluate(x_test_reshaped, y_test)
# print(f'Test Loss: {test_loss}')
# print(f'Test Accuracy: {test_accuracy}')

# # Plot training and validation accuracy
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
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

# # Save the HI-ResNet model weights
# resnet_model.save_weights('resnet_model_soft_voting2.weights.h5')
# print("ResNet model weights saved successfully.")

# # Create feature extraction
# feature_extractor = Model(inputs=resnet_model.input, outputs=resnet_model.layers[-4].output)

# # Extract features
# train_features = feature_extractor.predict(x_train_reshaped)
# test_features = feature_extractor.predict(x_test_reshaped)

# print('Shape of train features:', train_features.shape)
# print('Shape of test features:', test_features.shape)

# # Initialize ML classifiers
# svm_clf = SVC(probability=True)
# knn_clf = KNeighborsClassifier(n_neighbors=10)
# rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
# dt_clf = DecisionTreeClassifier(random_state=42)
# xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
# nb_clf = GaussianNB()

# # Train and evaluate ML classifiers
# classifiers = {'SVM': svm_clf, 'KNN': knn_clf, 'Random Forest': rf_clf, 'Decision Tree': dt_clf, 'XGBoost': xgb_clf, 'Naive Bayes': nb_clf}
# classifier_accuracies = {}

# for name, clf in classifiers.items():
#     clf.fit(train_features, y_train)
#     pred = clf.predict(test_features)
#     accuracy = accuracy_score(y_test, pred)
#     classifier_accuracies[name] = accuracy
#     print(f"{name} Accuracy:", accuracy)
#     print(classification_report(y_test, pred, target_names=CLASSES_LIST))

#     # Save each trained ML classifier
#     joblib.dump(clf, f'{name}_model.pkl')
#     print(f"{name} model saved successfully.")

#     # Confusion matrix of each ML classifier
#     conf_matrix = confusion_matrix(y_test, pred)
#     conf_matrix_normalized = conf_matrix.astype(float) / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

#     # Plot confusion matrix of each ML classifier
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title(f'Confusion Matrix - {name}')
#     plt.show()

# # Soft Voting Ensemble
# def soft_voting(classifiers, X):
#     proba_sum = np.zeros((X.shape[0], len(CLASSES_LIST)))
    
#     for name, clf in classifiers.items():
#         proba = clf.predict_proba(X)
#         proba_sum += proba
    
#     avg_proba = proba_sum / len(classifiers)
#     ensemble_pred = np.argmax(avg_proba, axis=1)
    
#     return ensemble_pred

# # Get predictions from the ensemble
# ensemble_pred = soft_voting(classifiers, test_features)

# # Evaluate the ensemble
# ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
# print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
# print(classification_report(y_test, ensemble_pred, target_names=CLASSES_LIST))

# # Confusion matrix for ensemble
# conf_matrix_ensemble = confusion_matrix(y_test, ensemble_pred)
# conf_matrix_normalized_ensemble = conf_matrix_ensemble.astype(float) / conf_matrix_ensemble.sum(axis=1)[:, np.newaxis] * 100

# # Plot confusion matrix for ensemble
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix_normalized_ensemble, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix - Ensemble')
# plt.show()


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, 
    Dropout, Dense, Input, Add, Activation, GlobalAveragePooling2D
)
from keras.models import Model
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import time
import joblib

# GPU Configuration
tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define Class
CLASSES_LIST = ['hugging', 'pushing', 'pointing', 'kicking', 'none-interaction']

# Load data
X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_train.npy")
X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_test.npy")

# Reshape data for input 
x_train_reshaped = np.reshape(X_train, (X_train.shape[0], 51, 50, 2))
x_test_reshaped = np.reshape(X_test, (X_test.shape[0], 51, 50, 2))
print("x_train_reshaped: ", x_train_reshaped.shape)

# Input shape format
input_shape = (51, 50, 2)

# Define the HI-ResNet block
def resnet_block(inputs, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(inputs)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

inputs = Input(shape=input_shape)

# Create HI-ResNet model
x = resnet_block(inputs, filters=32, kernel_size=(3, 3), strides=1)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

x = resnet_block(x, filters=64, kernel_size=(3, 3), strides=1)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

x = resnet_block(x, filters=128, kernel_size=(3, 3), strides=1)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)

x = Dense(100, activation='relu')(x)
x = Dropout(0.2)(x)

x = Dense(100, activation='relu')(x)
x = Dropout(0.2)(x)

outputs = Dense(len(CLASSES_LIST), activation='softmax')(x)
resnet_model = Model(inputs, outputs)

resnet_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
resnet_model.summary()

# Track the start training time
start_time = time.time()

# Train the HI-ResNet model
history = resnet_model.fit(x_train_reshaped, y_train, epochs=300, batch_size=4, shuffle=True, validation_data=(x_test_reshaped, y_test))

# Track the end training time
end_time = time.time()
training_time = end_time - start_time
print(f'Total Training Time: {training_time:.2f} seconds')

# Evaluate the model
test_loss, test_accuracy = resnet_model.evaluate(x_test_reshaped, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

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

# Save the HI-ResNet model weights
resnet_model.save_weights('resnet_model_soft_voting2.weights.h5')
print("ResNet model weights saved successfully.")

# Create feature extraction
feature_extractor = Model(inputs=resnet_model.input, outputs=resnet_model.layers[-4].output)

# Extract features
train_features = feature_extractor.predict(x_train_reshaped)
test_features = feature_extractor.predict(x_test_reshaped)

print('Shape of train features:', train_features.shape)
print('Shape of test features:', test_features.shape)

# Initialize ML classifiers
svm_clf = SVC(probability=True)
knn_clf = KNeighborsClassifier(n_neighbors=10)
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
dt_clf = DecisionTreeClassifier(random_state=42)
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
nb_clf = GaussianNB()

# Train and evaluate ML classifiers
classifiers = {'SVM': svm_clf, 'KNN': knn_clf, 'Random Forest': rf_clf, 'Decision Tree': dt_clf, 'XGBoost': xgb_clf, 'Naive Bayes': nb_clf}
classifier_accuracies = {}

for name, clf in classifiers.items():
    clf.fit(train_features, y_train)
    pred = clf.predict(test_features)
    accuracy = accuracy_score(y_test, pred)
    classifier_accuracies[name] = accuracy
    print(f"{name} Accuracy:", accuracy)
    print(classification_report(y_test, pred, target_names=CLASSES_LIST))

    # Save each trained ML classifier
    joblib.dump(clf, f'{name}_model.pkl')
    print(f"{name} model saved successfully.")

    # Confusion matrix of each ML classifier
    conf_matrix = confusion_matrix(y_test, pred)
    conf_matrix_normalized = conf_matrix.astype(float) / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    # Plot confusion matrix of each ML classifier
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

# Hard Voting Ensemble
def hard_voting(classifiers, X):
    predictions = np.zeros((len(classifiers), X.shape[0]), dtype=int)
    
    for i, (name, clf) in enumerate(classifiers.items()):
        pred = clf.predict(X)
        predictions[i, :] = pred
    
    # Mode of predictions
    from scipy.stats import mode
    ensemble_pred, _ = mode(predictions, axis=0)
    
    return ensemble_pred.flatten()

# Get predictions from the ensemble
ensemble_pred = hard_voting(classifiers, test_features)

# Evaluate the ensemble
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
print(classification_report(y_test, ensemble_pred, target_names=CLASSES_LIST))

# Confusion matrix for ensemble
conf_matrix_ensemble = confusion_matrix(y_test, ensemble_pred)
conf_matrix_normalized_ensemble = conf_matrix_ensemble.astype(float) / conf_matrix_ensemble.sum(axis=1)[:, np.newaxis] * 100

# Plot confusion matrix for ensemble
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_normalized_ensemble, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Ensemble')
plt.show()

