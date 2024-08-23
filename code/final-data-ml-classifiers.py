# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# import seaborn as sns
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier

# CLASSES_LIST = ['hugging', 'pushing', 'pointing', 'kicking', 'none-interaction']

# # Load data
# X_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_train.npy")
# y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_train.npy")
# X_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_test.npy")
# y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_test.npy")

# # Print shapes to debug
# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)

# # Check the number of dimensions
# n_train_samples, *dims = X_train.shape
# n_test_samples = X_test.shape[0]
# X_train_flattened = X_train.reshape(n_train_samples, -1)
# X_test_flattened = X_test.reshape(n_test_samples, -1)

# print("Flattened X_train shape:", X_train_flattened.shape)
# print("Flattened X_test shape:", X_test_flattened.shape)

# # Initialize classifiers
# svm_clf = SVC()
# knn_clf = KNeighborsClassifier(n_neighbors=20)
# rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# # List of classifiers
# classifiers = {'SVM': svm_clf, 'KNN': knn_clf, 'Random Forest': rf_clf}

# for name, clf in classifiers.items():
#     # Train the classifier
#     clf.fit(X_train_flattened, y_train)
    
#     # Predict on the test set
#     y_pred = clf.predict(X_test_flattened)
    
#     # Calculate accuracy
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"{name} Test Accuracy: {accuracy:.4f}")
    
#     # Confusion matrix and classification report
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     conf_matrix_normalized = conf_matrix.astype(float) / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

#     class_report = classification_report(y_test, y_pred, target_names=CLASSES_LIST)
#     print(f"{name} Classification Report:")
#     print("===========================================================")
#     print(class_report)
#     print("===========================================================")

#     # Plot confusion matrix
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title(f'Confusion Matrix - {name}')
#     plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

CLASSES_LIST = ['hugging', 'pushing', 'pointing', 'kicking', 'none-interaction']

X_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_train.npy")
y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_train.npy")
X_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_test.npy")
y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_test.npy")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Check the number of dimensions
n_train_samples, *dims = X_train.shape
n_test_samples = X_test.shape[0]
X_train_flattened = X_train.reshape(n_train_samples, -1)
X_test_flattened = X_test.reshape(n_test_samples, -1)

print("Flattened X_train shape:", X_train_flattened.shape)
print("Flattened X_test shape:", X_test_flattened.shape)

# Normalize the data
scaler = StandardScaler()
X_train_flattened = scaler.fit_transform(X_train_flattened)
X_test_flattened = scaler.transform(X_test_flattened)

# Define an expanded parameter grid for SVM
param_grid_svm = {
    'C': [0.1, 1.0, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.001, 0.0001],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [3, 4, 5], 
    'coef0': [0.0, 0.1, 0.5, 1.0] 
}

svm_clf = SVC()

# Apply Grid Search with Cross-Validation
grid_search_svm = GridSearchCV(svm_clf, param_grid_svm, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search_svm.fit(X_train_flattened, y_train)


best_svm_clf = grid_search_svm.best_estimator_
print(f"Best parameters for SVM: {grid_search_svm.best_params_}")

y_pred_svm = best_svm_clf.predict(X_test_flattened)


accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Test Accuracy: {accuracy_svm:.4f}")

conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
conf_matrix_normalized_svm = conf_matrix_svm.astype(float) / conf_matrix_svm.sum(axis=1)[:, np.newaxis] * 100

class_report_svm = classification_report(y_test, y_pred_svm, target_names=CLASSES_LIST)
print(f"SVM Classification Report:")
print("===========================================================")
print(class_report_svm)
print("===========================================================")

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_normalized_svm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix - SVM')
plt.show()
