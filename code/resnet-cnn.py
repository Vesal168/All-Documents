
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, 
    Dropout, Dense, Input, Add, Activation, GlobalAveragePooling2D, concatenate
)
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
import time

tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CLASSES_LIST = ['hugging', 'pushing', 'pointing', 'kicking', 'none-interaction']

# Load data
X_train, y_train = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_train.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_train.npy")
X_test, y_test = np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\X_test.npy"), np.load(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\data-preparation\split-data-80-20-5-110video\y_test.npy")

# Reshape data
x_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 50, 2))
x_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 50, 2))

print("x_train_reshaped: ", x_train_reshaped.shape)

input_shape = (51, 50, 2)

# Define the ResNet block
def resnet_block(inputs, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same', kernel_regularizer=l2(0.01))(inputs)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

# Create ResNet feature extractor
inputs_resnet = Input(shape=input_shape)
x = resnet_block(inputs_resnet, filters=32, kernel_size=(3, 3), strides=1)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.5)(x)

x = resnet_block(x, filters=64, kernel_size=(3, 3), strides=1)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.5)(x)

x = resnet_block(x, filters=128, kernel_size=(3, 3), strides=1)
x = GlobalAveragePooling2D()(x)

resnet_feature_extractor = Model(inputs_resnet, x)

# Define the second CNN model
inputs_cnn = Input(shape=input_shape)
y = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01))(inputs_cnn)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = Dropout(0.5)(y)

y = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01))(y)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = Dropout(0.5)(y)

y = Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01))(y)
y = GlobalAveragePooling2D()(y)

cnn_feature_extractor = Model(inputs_cnn, y)

# Extract features
train_features_resnet = resnet_feature_extractor.predict(x_train_reshaped)
test_features_resnet = resnet_feature_extractor.predict(x_test_reshaped)

train_features_cnn = cnn_feature_extractor.predict(x_train_reshaped)
test_features_cnn = cnn_feature_extractor.predict(x_test_reshaped)

# Normalize the features
from sklearn.preprocessing import StandardScaler
scaler_resnet = StandardScaler()
scaler_cnn = StandardScaler()

train_features_resnet = scaler_resnet.fit_transform(train_features_resnet)
test_features_resnet = scaler_resnet.transform(test_features_resnet)

train_features_cnn = scaler_cnn.fit_transform(train_features_cnn)
test_features_cnn = scaler_cnn.transform(test_features_cnn)

# Combine features
train_features_combined = np.concatenate((train_features_resnet, train_features_cnn), axis=1)
test_features_combined = np.concatenate((test_features_resnet, test_features_cnn), axis=1)

# Define the combined model
combined_inputs = Input(shape=(train_features_combined.shape[1],))
z = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(combined_inputs)
z = Dropout(0.5)(z)
z = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(z)
z = Dropout(0.5)(z)
z = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(z)
z = Dropout(0.5)(z)
outputs = Dense(len(CLASSES_LIST), activation='softmax')(z)

combined_model = Model(combined_inputs, outputs)

combined_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

combined_model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# Track the training time
start_time = time.time()

# Train the combined model
history = combined_model.fit(train_features_combined, y_train, epochs=300, batch_size=16, shuffle=True, validation_data=(test_features_combined, y_test), callbacks=[early_stopping, reduce_lr])

end_time = time.time()
training_time = end_time - start_time
print(f'Total Training Time: {training_time:.2f} seconds')

# Evaluate the model
test_loss, test_accuracy = combined_model.evaluate(test_features_combined, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Predict and evaluate
y_pred = combined_model.predict(test_features_combined)
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









