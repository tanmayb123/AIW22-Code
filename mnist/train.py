import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def standardize(data, axis):
    """Standardize the data in the given NumPy array along the specified axis."""
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return (data - mean) / std, mean, std


def render_normal_image(array):
    return Image.fromarray(array[:, :, 0])


def render_standardized_image(array):
    data = (array / np.abs(array).max() + 1) / 2
    data = (data * 255).astype(np.uint8)
    return Image.fromarray(data[:, :, 0])


def load_mnist(render=True):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    if render:
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)
        render_normal_image(X_train[0]).show()

    X_train, mean, std = standardize(X_train, (0, 1, 2))
    X_test = (X_test - mean) / std

    if render:
        render_standardized_image(X_train[0]).show()

    return X_train, y_train, X_test, y_test


class MNISTModel(Model):
    def __init__(self):
        super().__init__()

        self.convolutions = tf.keras.models.Sequential(
            [
                Conv2D(32, kernel_size=5, activation=tf.nn.relu),
                Conv2D(64, kernel_size=5, activation=tf.nn.relu),
                Conv2D(64, kernel_size=3, activation=tf.nn.relu),
                Conv2D(128, kernel_size=3, activation=tf.nn.relu),
                Conv2D(128, kernel_size=3, activation=tf.nn.relu),
                Conv2D(256, kernel_size=3, activation=tf.nn.relu),
                Conv2D(256, kernel_size=3, activation=tf.nn.relu),
            ]
        )

        self.gap = GlobalAveragePooling2D()
        self.classes = Dense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        x = self.convolutions(inputs)
        x = self.gap(x)                  # mean    (batch_size, 256)
        # x = torch.view(x, x.size(0), -1) # flatten (batch_size, 10 * 10 * 256)
        x = self.classes(x)
        return x

    def embed(self, inputs):
        x = self.convolutions(inputs)
        x = self.gap(x)
        return x


def main():
    X_train, y_train, X_test, y_test = load_mnist()

    model = MNISTModel()
    model.build(input_shape=(None, 28, 28, 1))
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=128
    )
    model.save_weights("mnist_model.h5")


if __name__ == "__main__":
    main()
