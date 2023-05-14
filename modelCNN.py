import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json

# PMDAS, arabic numbers, english alphabet


width = 28
height = 28

training_data = "/samp_data"
testing_data = ""
validation_data = ""

data_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2,
                              zoom_range=0.2, horizontal_flip=True)

train_dataset = data_gen.flow_from_directory(training_data, target_size=(
    32, 32), color_mode="grayscale", batch_size=15, subset='training')
test_dataset = data_gen.flow_from_directory(testing_data, target_size=(
    32, 32), color_mode="grayscale", batch_size=15, subset='training')
validation_dataset = data_gen.flow_from_directory(validation_data, target_size=(
    32, 32), color_mode="grayscale", batch_size=15, subset='training')

# Classes will be automatically inferred from the subdirectory names/structure
# under directory, where each subdirectory will be treated as a different class
train_labels = train_dataset.class_indices
num_labels = len(train_labels)

with open('label_map.json', 'r') as json_file:
    label_map = json.load(json_file)

model = Sequential([
    # Convolution with
    Conv2D(filters=24, kernel_size=(4, 4), strides=(1, 1),
           padding='same', input_shape=(28, 28, 1), activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Conv2D(filters=48, kernel_size=(5, 5), strides=(
        1, 1), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Conv2D(filters=64, kernel_size=(5, 5), strides=(
        1, 1), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Flatten(),
    Dense(units=128, activation='relu'),
    Dense(units=84, activation='relu'),
    Dense(units=num_labels, activation='softmax')
])

# The Adam optimizer is used to update the model's weights, categorical
# cross-entropy is chosen as the loss function to optimize, and accuracy is
# selected as the metric to evaluate the model's performance.
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, num_labels, batch_size=32, epochs=10)

model.save('ocrCNN.h5')

# Evaluate with model.evaluate(valadation)
