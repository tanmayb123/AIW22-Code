import numpy as np

#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

from train import load_data

def load_model():
    model = ResNet50V2(weights="imagenet")
    feature_layer = model.layers[-2]

    for layer in model.layers:
        layer.trainable = False

    probabilities_layer = Dense(120, activation="softmax")(feature_layer.output)
    model = Model(inputs=model.input, outputs=probabilities_layer)

    return model

def load_pretrained_model():
    loaded_model = load_model()
    loaded_model.load_weights('resnet50v2_classifier.h5')
    return loaded_model

def main():
    (train_images, train_labels), (test_images, test_labels) = load_data(use_existing_shuffle=True, preprocessing=True)

    model = load_pretrained_model()
    teacher_labels = model.predict(train_images, verbose=1, batch_size=256)

    np.save('teacher_labels.npy', teacher_labels)

if __name__ == "__main__":
    main()
