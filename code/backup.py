
# import os
# import numpy as np
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
# os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES']= '2'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# total_sample = 528
# total_frames = 51
# keypoints = 50
# CLASSES_LIST = ['hugging', 'kicking', 'pushing', 'punching', 'pointing', 'none-interaction']

# # Load your data
# X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data\y_train.npy")
# X_valid, y_valid = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data\X_valid.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data\y_valid.npy")
# X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data\y_test.npy")

# # Convert input data to float32
# X_train = X_train.astype(np.float32)
# X_valid = X_valid.astype(np.float32)
# X_test = X_test.astype(np.float32)

# # Reshape your data to match the model's input shape
# x_train_reshaped = np.reshape(X_train, (total_sample, total_frames, keypoints, 2, 1)).astype(np.float32)
# x_valid_reshaped = np.reshape(X_valid, (61, total_frames, keypoints, 2, 1)).astype(np.float32)
# x_test_reshaped = np.reshape(X_test, (61, total_frames, keypoints, 2, 1)).astype(np.float32)

# # Define label encoding function
# def encode_labels(labels, classes_list):
#     label_dict = {label: i for i, label in enumerate(classes_list)}
#     encoded_labels = [label_dict[label] for label in labels]
#     return np.array(encoded_labels)

# # Encode labels
# y_train_encoded = encode_labels(y_train, CLASSES_LIST)
# y_valid_encoded = encode_labels(y_valid, CLASSES_LIST)
# y_test_encoded = encode_labels(y_test, CLASSES_LIST)

# # Create your model (old)
# # model = Sequential([
# #     Conv3D(32, kernel_size=(3, 3, 3), input_shape=(total_frames, keypoints, 2, 1), activation='relu', padding='same'),
# #     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
# #     Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
# #     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
# #     Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
# #     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
# #     Flatten(),
# #     Dense(128, activation='relu'),
# #     Dropout(0.5),
# #     Dense(len(CLASSES_LIST), activation='softmax')
# # ])
# # Create your model
# model = Sequential([
#     Conv3D(32, kernel_size=(3, 3, 3), input_shape=(total_frames, keypoints, 2, 1), activation='relu', padding='same'),
#     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
#     Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
#     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
#     Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
#     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dropout(0.3), 
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(len(CLASSES_LIST), activation='softmax')
# ])

# # Compile the model
# model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0005, decay=1e-6), metrics=['accuracy'])

# # Print model summary
# model.summary()

# # Train the model
# history = model.fit(x_train_reshaped, y_train_encoded, epochs=150, batch_size=32, shuffle=True, validation_data=(x_valid_reshaped, y_valid_encoded))


# # # Convert input data to float32
# # X_train = X_train.astype(np.float32)
# # X_valid = X_valid.astype(np.float32)
# # X_test = X_test.astype(np.float32)

# # # Reshape your data to match the model's input shape
# # x_train_reshaped = np.reshape(X_train, (total_sample, total_frames, keypoints, 2, 1)).astype(np.float32)
# # x_valid_reshaped = np.reshape(X_valid, (66, total_frames, keypoints, 2, 1)).astype(np.float32)
# # x_test_reshaped = np.reshape(X_test, (66, total_frames, keypoints, 2, 1)).astype(np.float32)

# # # Define label encoding function
# # def encode_labels(labels, classes_list):
# #     label_dict = {label: i for i, label in enumerate(classes_list)}
# #     encoded_labels = [label_dict[label] for label in labels]
# #     return np.array(encoded_labels)

# # # Encode labels
# # y_train_encoded = encode_labels(y_train, CLASSES_LIST)
# # y_valid_encoded = encode_labels(y_valid, CLASSES_LIST)
# # y_test_encoded = encode_labels(y_test, CLASSES_LIST)

# # # Create your model
# # model = Sequential([
# #     Conv3D(32, kernel_size=(3, 3, 3), input_shape=(total_frames, keypoints, 2, 1), activation='relu', padding='same'),
# #     BatchNormalization(),
# #     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
    
# #     Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
# #     BatchNormalization(),
# #     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
    
# #     Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
# #     BatchNormalization(),
# #     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
    
# #     Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same'),
# #     BatchNormalization(),
# #     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
    
# #     Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same'),
# #     BatchNormalization(),
# #     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
    
# #     Flatten(),
# #     Dense(256, activation='relu'),
# #     Dropout(0.5),
# #     Dense(len(CLASSES_LIST), activation='softmax')
# # ])

# # model = Sequential([
# #     Conv3D(32, kernel_size=(3, 3, 3), input_shape=(total_frames, keypoints, 2, 1), activation='relu', padding='same'),
# #     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
# #     Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
# #     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
# #     Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
# #     MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
# #     Flatten(),
# #     Dense(256, activation='relu'),
# #     Dropout(0.3), 
# #     Dense(128, activation='relu'),
# #     Dropout(0.3),
# #     Dense(len(CLASSES_LIST), activation='softmax')
# # ])

# # # Compile the model
# # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # # Print model summary
# # model.summary()

# # # Train the model
# # history = model.fit(x_train_reshaped, y_train_encoded, epochs=100, batch_size=4, shuffle=True, validation_data=(x_valid_reshaped, y_valid_encoded))

# test_loss, test_accuracy = model.evaluate(x_test_reshaped, y_test_encoded)
# print(f'Test Loss: {test_loss}')
# print(f'Test Accuracy: {test_accuracy}')

# y_pred = model.predict(x_test_reshaped)
# y_pred_classes = np.argmax(y_pred, axis=1)

# conf_matrix = confusion_matrix(y_test_encoded, y_pred_classes)
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()












# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
import lightgbm as lgb
# from sklearn.tree import DecisionTreeClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


total_sample = 385
total_frames = 51 
keypoints = 50
CLASSES_LIST = ['hugging', 'kicking', 'pushing', 'pointing', 'none-interaction']

# # Load your data
# X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_train.npy")
# X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_test.npy")

# X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\test-lg-te\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\test-lg-te\y_train.npy")
# X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\test-lg-te\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\test-lg-te\y_test.npy")
X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\y_train.npy")
X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\y_test.npy")


# X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data-interpolate\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data-interpolate\y_train.npy")
# X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data-interpolate\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data-interpolate\y_test.npy")

# # Convert input data to float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# # Reshape your data to match the model's input shape
x_train_reshaped = np.reshape(X_train, (total_sample, total_frames, keypoints, 2, 1)).astype(np.float32)
x_test_reshaped = np.reshape(X_test, (165, total_frames, keypoints, 2, 1)).astype(np.float32)

# # Reshape your data to 2D array (flatten)
x_train_reshaped_flatten = x_train_reshaped.reshape(x_train_reshaped.shape[0], -1)
x_test_reshaped_flatten = x_test_reshaped.reshape(x_test_reshaped.shape[0], -1)

# # Define label encoding function
# def encode_labels(labels, classes_list):
#     label_dict = {label: i for i, label in enumerate(classes_list)}
#     encoded_labels = [label_dict[label] for label in labels]
#     return np.array(encoded_labels)

# # Encode labels
# y_train_encoded = encode_labels(y_train, CLASSES_LIST)
# y_test_encoded = encode_labels(y_test, CLASSES_LIST)

# # Initialize KNN classifier
# knn = KNeighborsClassifier(n_neighbors=2)
# rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
# svm_classifier = SVC(kernel='rbf', random_state=42)
lgb_classifier = lgb.LGBMClassifier()
# dt_classifier = DecisionTreeClassifier()



# # Train the KNN classifier
# knn.fit(x_train_reshaped_flatten, y_train)
# rf_classifier.fit(x_train_reshaped_flatten, y_train)
# svm_classifier.fit(x_train_reshaped_flatten, y_train)
lgb_classifier.fit(x_train_reshaped_flatten, y_train)
# dt_classifier.fit(x_train_reshaped_flatten, y_train)


# # Predict labels for test data
# y_pred_knn = knn.predict(x_test_reshaped_flatten)
# y_pred_rf = rf_classifier.predict(x_test_reshaped_flatten)
# y_pred_svm = svm_classifier.predict(x_test_reshaped_flatten)
y_pred_lgb = lgb_classifier.predict(x_test_reshaped_flatten)
# y_pred_dt = dt_classifier.predict(x_test_reshaped_flatten)



# # Evaluate accuracy
# accuracy_knn = np.mean(y_pred_knn == y_test)
# print(f'KNN Accuracy: {accuracy_knn}')

# accuracy_rf = np.mean(y_pred_rf == y_test)
# print(f'Random Forest Accuracy: {accuracy_rf}')

# accuracy_knn = accuracy_score(y_test, y_pred_knn)
# classification_report_knn = classification_report(y_test, y_pred_knn)

# print("RF Classification Report:\n", classification_report_knn)

# print("Random forest Accuracy:", accuracy_rf)
# print("KNN Classification Report:\n", classification_report_rf)

# accuracy_svm = np.mean(y_pred_svm == y_test)
# print(f'SVM Accuracy: {accuracy_svm}')

accuracy_lgb = np.mean(y_pred_lgb == y_test)
print(f'LightGBM Accuracy: {accuracy_lgb}')

accuracy_lightGBM = accuracy_score(y_test, y_pred_lgb)
classification_report_lightGBM = classification_report(y_test, y_pred_lgb)

print(" Classification Report:\n", classification_report_lightGBM)

# accuracy_dt = np.mean(y_pred_dt == y_test)
# print(f'Decision Tree Accuracy: {accuracy_dt}')

# # Calculate and plot confusion matrix

# conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix (KNN)')
# plt.show()

conf_matrix_rf = confusion_matrix(y_test, y_pred_lgb)
conf_matrix = conf_matrix_rf / conf_matrix_rf.astype(float).sum(axis=1) * 100
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Random Forest)')
plt.show()

# conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix (SVM)')
# plt.show()

# # conf_matrix_lgb = confusion_matrix(y_test_encoded, y_pred_lgb)
# # plt.figure(figsize=(10, 8))
# # sns.heatmap(conf_matrix_lgb, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
# # plt.xlabel('Predicted Labels')
# # plt.ylabel('True Labels')
# # plt.title('Confusion Matrix (LightGBM)')
# # plt.show()
# # conf_matrix_dt = confusion_matrix(y_test_encoded, y_pred_dt)
# # plt.figure(figsize=(10, 8))
# # sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
# # plt.xlabel('Predicted Labels')
# # plt.ylabel('True Labels')
# # plt.title('Confusion Matrix (Decision Tree)')
# # plt.show()



#     Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'),
#     Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    
#     Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
#     Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    
#     Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
#     Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
#     Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    
#     Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
#     Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
#     Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    
#     Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
#     Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
#     Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    
#     Flatten(),
#     Dense(4096, activation='relu'),
#     Dense(4096, activation='relu'),
#     Dense(len(CLASSES_LIST), activation='softmax')
# ])







# #CNN + LSTM


# import os
# import numpy as np
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, LSTM, TimeDistributed, Conv1D, MaxPooling1D
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Reshape, GlobalAveragePooling2D, BatchNormalization

# tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
# os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES']= '2'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CLASSES_LIST = ['hugging', 'kicking', 'pushing', 'punching', 'pointing', 'none-interaction']


# X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_train.npy")
# X_val, y_val = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_valid.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_valid.npy")

# x_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 50, 2))
# x_val_reshaped = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 50, 2))

# print("x_train_reshaped: ", x_train_reshaped.shape)

# input_shape = (51, 50, 2)
# # Define the model

# model = Sequential([ 
#     Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),

#     Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),
    
#     Reshape((11, 704)),  # Reshape for LSTM, (11, 704)

#     LSTM(100, return_sequences=True, input_shape= (11, 704), activation='relu'),
#     Dropout(0.3),

#     LSTM(100, return_sequences=True, activation='relu'),
#     Dropout(0.3),

#     LSTM(100, return_sequences=True, activation='relu'),
#     Dropout(0.3),

#     LSTM(200, return_sequences=False, activation='relu'),
#     Dropout(0.3),

#     Dense(100, activation='relu'),
#     Dropout(0.5),

#     Dense(len(CLASSES_LIST), activation='softmax')
# ])


# # Compile the model
# model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001, decay=1e-6), metrics=['accuracy']) #sparse_categorical_crossentropy

# # Print model summary
# model.summary()

# # Train the model
# history = model.fit(x_train_reshaped, y_train, epochs=250, batch_size=32, shuffle=True, validation_data=(x_val_reshaped, y_val))

# #==============================test model =============================
# X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_test.npy")

# # Reshape X_test if needed
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 50, 2))

# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(X_test, y_test)

# print(f'Test Loss: {test_loss}')
# print(f'Test Accuracy: {test_accuracy}')

# # Predict classes for X_test
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)

# # Create confusion matrix
# conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()

# # Plot training and validation loss
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['loss'], label='Training Loss')
# # Assuming you have validation data and you monitored validation loss during training
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()



