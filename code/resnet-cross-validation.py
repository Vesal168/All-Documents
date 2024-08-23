
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import time

# GPU Configuration
tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CLASSES_LIST = ['hugging', 'pushing', 'pointing', 'kicking', 'none-interaction']

# Load data
X, y = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_train.npy")

# Reshape data for ResNet
X_reshaped = np.reshape(X, (X.shape[0], 51, 50, 2))

input_shape = (51, 50, 2)

# Define the ResNet block
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

# Function to create ResNet model
def create_resnet_model(input_shape):
    inputs = Input(shape=input_shape)

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

    model = Model(inputs, outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    
    return model

# K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

fold = 1
all_fold_results = []

for train_index, val_index in kf.split(X_reshaped):
    print(f"Fold {fold}")
    X_train_fold, X_val_fold = X_reshaped[train_index], X_reshaped[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    # Create and train ResNet model
    resnet_model = create_resnet_model(input_shape)
    start_time = time.time()
    resnet_model.fit(X_train_fold, y_train_fold, epochs=200, batch_size=4, shuffle=True, validation_data=(X_val_fold, y_val_fold))
    end_time = time.time()
    training_time = end_time - start_time
    print(f'Total Training Time for Fold {fold}: {training_time:.2f} seconds')

    # Feature extraction
    feature_extractor = Model(inputs=resnet_model.input, outputs=resnet_model.layers[-4].output)
    train_features = feature_extractor.predict(X_train_fold)
    val_features = feature_extractor.predict(X_val_fold)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=50)
    train_features_pca = pca.fit_transform(train_features)
    val_features_pca = pca.transform(val_features)

    # Initialize classifiers
    svm_clf = SVC()
    knn_clf = KNeighborsClassifier(n_neighbors=10)
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)

    classifiers = {'SVM': svm_clf, 'KNN': knn_clf, 'Random Forest': rf_clf}
    classifier_accuracies = {}

    for name, clf in classifiers.items():
        clf.fit(train_features_pca, y_train_fold)
        pred = clf.predict(val_features_pca)
        accuracy = accuracy_score(y_val_fold, pred)
        classifier_accuracies[name] = accuracy
        print(f"{name} Accuracy for Fold {fold}:", accuracy)
        print(classification_report(y_val_fold, pred, target_names=CLASSES_LIST))

    # Sort classifiers by accuracy and select the top 3
    sorted_classifiers = sorted(classifier_accuracies.items(), key=lambda item: item[1], reverse=True)
    top_three_classifiers = sorted_classifiers[:3]

    # Print the top three classifiers
    print(f"Top three classifiers for Fold {fold}:")
    for name, accuracy in top_three_classifiers:
        print(f"{name}: {accuracy:.4f}")

    # Create a dictionary of the top three classifiers
    top_classifiers = [(name, classifiers[name]) for name, _ in top_three_classifiers]

    # Create a voting classifier with the top three classifiers
    voting_clf = VotingClassifier(estimators=top_classifiers, voting='hard')

    # Train the voting classifier
    voting_clf.fit(train_features_pca, y_train_fold)

    # Evaluate the voting classifier
    voting_pred = voting_clf.predict(val_features_pca)
    voting_accuracy = accuracy_score(y_val_fold, voting_pred)
    print(f"Top Three Classifier Voting Accuracy for Fold {fold}:", voting_accuracy)
    print(classification_report(y_val_fold, voting_pred, target_names=CLASSES_LIST))

    # Record the results
    fold_results = {
        'fold': fold,
        'svm_accuracy': classifier_accuracies['SVM'],
        'knn_accuracy': classifier_accuracies['KNN'],
        'rf_accuracy': classifier_accuracies['Random Forest'],
        'voting_accuracy': voting_accuracy
    }
    all_fold_results.append(fold_results)

    fold += 1

# Print all fold results
print("\nAll fold results:")
for result in all_fold_results:
    print(result)
