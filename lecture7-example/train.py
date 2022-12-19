import numpy as np
import random

#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

TRAIN_PERCENT = 0.8

def load_model():
    model = ResNet50V2(weights="imagenet")
    feature_layer = model.layers[-2]

    for layer in model.layers:
        layer.trainable = False

    probabilities_layer = Dense(120, activation="softmax")(feature_layer.output)
    model = Model(inputs=model.input, outputs=probabilities_layer)

    model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

def load_data(use_existing_shuffle=False, preprocessing=True):
    images = np.load("images.npy")
    labels = np.load("labels.npy")

    if preprocessing:
        images = preprocess_input(images)

    if use_existing_shuffle:
        data_idxs = np.load("data_shuffle.npy")
    else:
        data_idxs = list(range(len(images)))
        random.shuffle(data_idxs)
        np.save("data_shuffle.npy", data_idxs)

    images = images[data_idxs]
    labels = labels[data_idxs]

    train_num = int(len(images) * TRAIN_PERCENT)

    train_images = images[:train_num]
    train_labels = labels[:train_num]
    test_images = images[train_num:]
    test_labels = labels[train_num:]

    return (train_images, train_labels), (test_images, test_labels)

def main():
    (train_images, train_labels), (test_images, test_labels) = load_data()

    print("Loaded data")
    print("Train images:", train_images.shape)
    print("Train labels:", train_labels.shape)
    print("Test images:", test_images.shape)
    print("Test labels:", test_labels.shape)

    model = load_model()

    model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

    model.save("resnet50v2_classifier.h5")

if __name__ == "__main__":
    main()
