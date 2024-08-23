

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

# CLASSES_LIST = ['hugging', 'kicking', 'pushing', 'pointing', 'none-interaction']

# # X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data\y_train.npy")
# # X_val, y_val = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data\X_valid.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data\y_valid.npy")

# #data after crop
# # X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_train.npy")
# # X_val, y_val = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_valid.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_valid.npy")

# X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\y_train.npy")
# X_val, y_val = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\y_test.npy")


# # Convert input data to float32
# #X_train = X_train.astype(np.float32)
# #X_test = X_test.astype(np.float32)

# # Reshape your data to match the model's input shape
# #x_train_reshaped = np.reshape(X_train, (total_sample, total_frames, keypoints, 2, 1)).astype(np.float32)
# #x_test_reshaped = np.reshape(X_test, (198, total_frames, keypoints, 2, 1)).astype(np.float32)


# #print("y_train: ", len(y_train))
# # from keras.utils import to_categorical

# x_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 50, 2))
# x_val_reshaped = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 50, 2))
# #y_train = to_categorical(y_train)
# #y_val =  to_categorical(y_val)


# print("x_train_reshaped: ", x_train_reshaped.shape)


# # Define label encoding function
# '''def encode_labels(labels, classes_list):
#     label_dict = {label: i for i, label in enumerate(classes_list)}
#     encoded_labels = [label_dict[label] for label in labels]
#     return np.array(encoded_labels)'''

# # Encode labels
# #y_train_encoded = encode_labels(y_train, CLASSES_LIST)
# #y_test_encoded = encode_labels(y_test, CLASSES_LIST)

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

#     LSTM(200, return_sequences=False, activation='relu'),
#     Dropout(0.3),

#     Dense(100, activation='relu'),
#     Dropout(0.3),

#     Dense(100, activation='relu'),
#     Dropout(0.3),

#     Dense(len(CLASSES_LIST), activation='softmax')
# ])


# # Compile the model
# model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001, decay=1e-6), metrics=['accuracy']) #sparse_categorical_crossentropy

# # Print model summary
# model.summary()

# # Train the model
# history = model.fit(x_train_reshaped, y_train, epochs=250, batch_size=32, shuffle=True, validation_data=(x_val_reshaped, y_val))


# #==============================test model =============================
# #X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_test.npy")

# # Reshape X_test if needed
# #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 50, 2))

# # Convert y_test to categorical format if needed
# #y_test = to_categorical(y_test)

# # Evaluate the model
# val_loss, val_accuracy = model.evaluate(x_val_reshaped, y_val)

# print(f'Test Loss: {val_loss}')
# print(f'Test Accuracy: {val_accuracy}')

# # Predict classes for X_test
# y_pred = model.predict(x_val_reshaped)
# y_pred_classes = np.argmax(y_pred, axis=1)

# conf_matrix = confusion_matrix(y_val, y_pred_classes)
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


#old code

# import os
# import numpy as np
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Reshape, BatchNormalization
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CLASSES_LIST = ['hugging', 'kicking', 'pushing', 'pointing', 'none-interaction']

# # Load data
# # X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_train.npy")
# # X_val, y_val = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_valid.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_valid.npy")
# # X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_test.npy")

# X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\y_train.npy")
# X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\y_test.npy")

# # Reshape input data
# x_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 50, 2))
# # x_val_reshaped = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 50, 2))
# x_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 50, 2))

# # Define the model
# model = Sequential([ 
#     Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(51, 50, 2)),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),

#     Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),
    
#     Reshape((11, 704)),  # Reshape for LSTM, (11, 704)

#     LSTM(100, return_sequences=True, activation='relu'),
#     Dropout(0.3),

#     LSTM(100, return_sequences=True, activation='relu'),
#     Dropout(0.3),

#     LSTM(100, return_sequences=True, activation='relu'),
#     Dropout(0.3),

#     LSTM(200, return_sequences=False, activation='relu'),
#     Dropout(0.3)
# ])

# # Compile the model
# model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001, decay=1e-6), metrics=['accuracy'])

# # Train the model
# model.fit(x_train_reshaped, y_train, epochs=250, batch_size=32, shuffle=True, validation_data=(x_test_reshaped, y_test))

# # Extract features
# train_features = model.predict(x_train_reshaped)
# # val_features = model.predict(x_val_reshaped)
# test_features = model.predict(x_test_reshaped)

# # Flatten features
# train_features_flat = train_features.reshape(train_features.shape[0], -1)
# # val_features_flat = val_features.reshape(val_features.shape[0], -1)
# test_features_flat = test_features.reshape(test_features.shape[0], -1)

# # Train RandomForestClassifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(train_features_flat, y_train)

# # Evaluate RandomForestClassifier
# train_acc = accuracy_score(y_train, rf_classifier.predict(train_features_flat))
# # val_acc = accuracy_score(y_val, rf_classifier.predict(val_features_flat))
# test_acc = accuracy_score(y_test, rf_classifier.predict(test_features_flat))

# print(f'Training Accuracy: {train_acc}')
# # print(f'Validation Accuracy: {val_acc}')
# print(f'Test Accuracy: {test_acc}')

# # Create confusion matrix
# conf_matrix = confusion_matrix(y_test, rf_classifier.predict(test_features_flat))
# plt.figure(figsize=(10, 8))
# sns.heatmap






import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, LSTM, TimeDistributed, Conv1D, MaxPooling1D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Reshape, LSTM, Dropout, Dense, Bidirectional,
    Conv1D, SpatialDropout1D
)
from keras.regularizers import l2


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Reshape, GlobalAveragePooling2D, BatchNormalization

tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']= '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CLASSES_LIST = ['hugging', 'kicking', 'pushing', 'pointing', 'none-interaction']

# X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data\y_train.npy")
# X_val, y_val = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data\X_valid.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-data\y_valid.npy")

#data after crop
# X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_train.npy")
# X_val, y_val = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_valid.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_valid.npy")

X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\y_train.npy")
X_val, y_val = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video-70-30-5classes\y_test.npy")
#X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-80-20-data-after-cropped-110video-4-classes\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-80-20-data-after-cropped-110video-4-classes\y_train.npy")
#X_val, y_val = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-80-20-data-after-cropped-110video-4-classes\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split-80-20-data-after-cropped-110video-4-classes\y_test.npy")


# Convert input data to float32
#X_train = X_train.astype(np.float32)
#X_test = X_test.astype(np.float32)

# Reshape your data to match the model's input shape
#x_train_reshaped = np.reshape(X_train, (total_sample, total_frames, keypoints, 2, 1)).astype(np.float32)
#x_test_reshaped = np.reshape(X_test, (198, total_frames, keypoints, 2, 1)).astype(np.float32)


#print("y_train: ", len(y_train))
# from keras.utils import to_categorical

x_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 50, 2))
x_val_reshaped = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 50, 2))
#y_train = to_categorical(y_train)
#y_val =  to_categorical(y_val)


print("x_train_reshaped: ", x_train_reshaped.shape)


# Define label encoding function
'''def encode_labels(labels, classes_list):
    label_dict = {label: i for i, label in enumerate(classes_list)}
    encoded_labels = [label_dict[label] for label in labels]
    return np.array(encoded_labels)'''

# Encode labels
#y_train_encoded = encode_labels(y_train, CLASSES_LIST)
#y_test_encoded = encode_labels(y_test, CLASSES_LIST)

input_shape = (51, 50, 2)
# Define the model

model = Sequential([ 
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    

    Reshape((11, 704)),  # Reshape for LSTM, (11, 704)

    LSTM(100, return_sequences=True, input_shape= (11, 704), activation='relu'),
    Dropout(0.3),

    LSTM(200, return_sequences=False, activation='relu'),
    Dropout(0.3),

    Dense(100, activation='relu'),
    Dropout(0.3),

    Dense(100, activation='relu'),
    Dropout(0.3),

    Dense(len(CLASSES_LIST), activation='softmax')
])

# model = Sequential([
#     Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),

#     Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),

#     Reshape((11, 704)),  # Reshape for LSTM, (11, 704)

#     LSTM(100, return_sequences=True, activation='relu', kernel_regularizer=l2(0.01)),
#     Dropout(0.3),

#     LSTM(200, return_sequences=True, activation='relu', kernel_regularizer=l2(0.01)),
#     Dropout(0.3),

#     LSTM(200, return_sequences=False, activation='relu', kernel_regularizer=l2(0.01)),
#     Dropout(0.3),

#     Dense(100, activation='relu'),
#     Dropout(0.3),

#     Dense(len(CLASSES_LIST), activation='softmax')
# ])


# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, decay=1e-6), metrics=['accuracy']) #sparse_categorical_crossentropy

# Print model summary
model.summary()

# Train the model
history = model.fit(x_train_reshaped, y_train, epochs=800, batch_size=4, shuffle=True, validation_data=(x_val_reshaped, y_val))


#==============================test model =============================
#X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\split_data_after_crop_video\y_test.npy")

# Reshape X_test if needed
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 50, 2))

# Convert y_test to categorical format if needed
#y_test = to_categorical(y_test)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(x_val_reshaped, y_val)

print(f'Test Loss: {val_loss}')
print(f'Test Accuracy: {val_accuracy}')

# Predict classes for X_test
y_pred = model.predict(x_val_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)

conf_matrix = confusion_matrix(y_val, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()









