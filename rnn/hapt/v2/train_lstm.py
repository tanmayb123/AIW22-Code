import tensorflow as tf
import numpy as np

from data import CLASSES, LoadedExperiment, load_windowed_dataset


def create_model() -> tf.keras.Model:
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Dense(
            32,
            activation="tanh",
            input_shape=(
                50,
                6,
            ),
        )
    )
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation="tanh")))
    model.add(tf.keras.layers.Dense(len(CLASSES), activation="softmax"))
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.summary()
    return model


def create_finetuning_model(base: tf.keras.Model) -> tf.keras.Model:
    base.trainable = False
    base_output = base.layers[-2].output
    output = tf.keras.layers.Dense(6, activation="softmax")(base_output)
    model = tf.keras.Model(inputs=base.input, outputs=output)
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), metrics=["accuracy"]
    )
    return model


def main():
    (train_windows, train_labels), (test_windows, test_labels) = load_windowed_dataset()
    model = create_model()
    model.fit(
        train_windows,
        train_labels,
        epochs=10,
        validation_data=(test_windows, test_labels),
        shuffle=True,
        batch_size=32,
    )

    finetuning_model = create_finetuning_model(model)

    smaller_classes = np.where(train_labels >= 6)
    train_windows = train_windows[smaller_classes]
    train_labels = train_labels[smaller_classes] - 6

    smaller_classes = np.where(test_labels >= 6)
    test_windows = test_windows[smaller_classes]
    test_labels = test_labels[smaller_classes] - 6

    finetuning_model.fit(
        train_windows,
        train_labels,
        epochs=100,
        validation_data=(test_windows, test_labels),
        shuffle=True,
        batch_size=32,
    )


if __name__ == "__main__":
    main()
