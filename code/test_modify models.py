
from tensorflow.keras.applications import ResNet50

# Load the pre-trained ResNet50 model
#model = ResNet50(weights='imagenet')  # Use weights='imagenet' to load pre-trained weights on ImageNet
#model.summary()

#read data

import numpy as np

data = np.load("train_features.npy")

for i in range(0, data.shape[0]):
    print("data shape: ", data[i].shape)
    print("data value:", data[i])
    print("----------------------------------------------------------------------\n")