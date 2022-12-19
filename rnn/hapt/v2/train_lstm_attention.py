import random
from typing import Union, Tuple, Optional

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from data import ACTIVITIES, CLASSES, LoadedExperiment, load_windowed_dataset


class LSTMAttentionModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.class_query = tf.keras.layers.Dense(32, activation="tanh")

        self.motion_embed = tf.keras.layers.Dense(32, activation="tanh")
        self.motion_lstm = tf.keras.layers.LSTM(
            32, activation="tanh", return_sequences=True
        )
        self.motion_key = tf.keras.layers.Dense(32, activation="tanh")
        self.motion_value = tf.keras.layers.Dense(32, activation="tanh")

        self.score = tf.keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor, return_attention: bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        motion = self.motion_embed(inputs)
        motion = self.motion_lstm(motion)

        motion_keys = self.motion_key(motion)
        motion_values = self.motion_value(motion)

        scores = []
        classes_relevance = []
        for classidx in range(len(CLASSES)):
            onehot_class = tf.one_hot(classidx, len(CLASSES))[None, None, :]
            class_query = self.class_query(onehot_class)

            motion_keys_norm = tf.math.reduce_sum(motion_keys**2, axis=-1) ** 0.5
            class_query_norm = tf.math.reduce_sum(class_query**2, axis=-1) ** 0.5
            relevance = tf.math.reduce_sum(motion_keys * class_query, axis=-1) / (
                motion_keys_norm * class_query_norm
            )
            relevance = tf.nn.softmax(relevance, axis=1)
            classes_relevance.append(relevance)

            weighted_values = relevance[:, :, None] * motion_values
            weighted_values = tf.reduce_sum(weighted_values, axis=1)

            score = self.score(weighted_values)
            scores.append(score)

        scores = tf.concat(scores, axis=1)
        scores = tf.nn.softmax(scores, axis=1)

        if return_attention:
            classes_relevance = tf.stack(classes_relevance)
            return scores, classes_relevance

        return scores


def create_model() -> tf.keras.Model:
    model = LSTMAttentionModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.build(input_shape=(None, 50, 6))
    model.summary()
    return model


def visualize_attention(x_sample: np.ndarray, y_sample: np.ndarray, model: tf.keras.Model):
    scores, attention = model(x_sample[None, :], return_attention=True)
    fig, axs = plt.subplots(2, max([3, len(CLASSES)]))

    clr = ["red"] * len(CLASSES)
    clr[y_sample] = "green"
    labels = [ACTIVITIES[x] for x in CLASSES]
    axs[0, 0].bar(range(len(CLASSES)), scores[0], color=clr)
    axs[0, 0].set_xticks(range(len(CLASSES)))
    axs[0, 0].set_xticklabels(labels, rotation=20)

    for i in range(3):
        axs[0, 1].plot(x_sample[:, i])
    for i in range(3):
        axs[0, 2].plot(x_sample[:, i + 3])

    for i in range(len(CLASSES)):
        axs[1, i].plot(attention[i, 0])

    plt.show()


def main():
    class_weight = {
        0: 1.,
        1: 1.,
        2: 1.,
        3: 1.,
        4: 1.,
        5: 1.,
        6: 10.,
        7: 10.,
        8: 10.,
        9: 10.,
        10: 10.,
        11: 10.,
    }
    (train_windows, train_labels), (test_windows, test_labels) = load_windowed_dataset()
    model = create_model()
    model.fit(
        train_windows,
        train_labels,
        epochs=10,
        validation_data=(test_windows, test_labels),
        shuffle=True,
        class_weight=class_weight,
        batch_size=64,
    )

    smaller_classes = np.where(test_labels >= 6)
    test_windows = test_windows[smaller_classes]
    test_labels = test_labels[smaller_classes]

    evaluation = model.evaluate(test_windows, test_labels, verbose=2)
    print(evaluation)

    for i in random.sample(range(len(test_windows)), 10):
        visualize_attention(test_windows[i], test_labels[i], model)


if __name__ == "__main__":
    main()
