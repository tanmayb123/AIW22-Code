import numpy as np
import random

import tensorflow as tf

from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda
from tensorflow.keras.optimizers import Adam

from train import load_data
from get_teacher_labels import load_pretrained_model

tf.config.run_functions_eagerly(True)

TRAIN_PERCENT = 0.8

def load_student_model():
    model = MobileNetV3Small(weights=None)
    feature_layer = model.layers[-4]

    #for layer in model.layers:
    #    layer.trainable = False

    probabilities_layer = Flatten()(feature_layer.output)
    probabilities_layer = Dense(120, activation="softmax")(probabilities_layer)
    model = Model(inputs=model.input, outputs=probabilities_layer, name="Student")

    model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

def main():
    (train_images, train_labels), (test_images, test_labels) = load_data(use_existing_shuffle=True, preprocessing=False)

    print("Loaded data")
    print("Train images:", train_images.shape)
    print("Train labels:", train_labels.shape)
    print("Test images:", test_images.shape)
    print("Test labels:", test_labels.shape)

    model = load_student_model()
    model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels), shuffle=True)

    model.save("mobilenet_classifier.h5")

if __name__ == "__main__":
    main()
