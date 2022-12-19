import tensorflow as tf

from data import CLASSES, WINDOW_SIZE, LoadedExperiment, load_windowed_dataset


class SelfAttentionLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.query = tf.keras.layers.Dense(32)
        self.key = tf.keras.layers.Dense(32)
        self.value = tf.keras.layers.Dense(32)

        self.ff = tf.keras.layers.Dense(32, activation="tanh")

    def call(self, inputs):
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)

        attention = tf.matmul(query, key, transpose_b=True)
        attention = tf.nn.softmax(attention, axis=-1)
        attention = tf.matmul(attention, value)

        return self.ff(attention)


class SelfAttentionModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.position_embed = tf.keras.layers.Dense(32, activation="tanh")
        self.classify_embed = tf.keras.layers.Dense(32, activation="tanh")
        self.motion_embed = tf.keras.layers.Dense(32, activation="tanh")

        self.self_attention = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dropout(0.3),
                SelfAttentionLayer(),
                tf.keras.layers.Dropout(0.3),
                SelfAttentionLayer(),
                tf.keras.layers.Dropout(0.3),
                SelfAttentionLayer(),
                tf.keras.layers.Dropout(0.3),
            ]
        )

        self.classifier = tf.keras.layers.Dense(len(CLASSES), activation="softmax")

    def call(self, inputs):
        position_embed = self.position_embed(tf.eye(WINDOW_SIZE))[None, :]

        classify_embed = self.classify_embed(tf.one_hot(0, 1)[None, None, :])
        classify_embed = tf.repeat(classify_embed, tf.shape(inputs)[0], axis=0)

        motion_embed = self.motion_embed(inputs) + position_embed

        input_seq = tf.concat([classify_embed, motion_embed], axis=1)

        attention = self.self_attention(input_seq)

        classify_token = attention[:, 0]

        return self.classifier(classify_token)


def create_model() -> tf.keras.Model:
    model = SelfAttentionModel()
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.build(input_shape=(None, WINDOW_SIZE, 6))
    model.summary()
    return model


def main():
    (train_windows, train_labels), (test_windows, test_labels) = load_windowed_dataset()
    model = create_model()
    model.fit(
        train_windows,
        train_labels,
        epochs=10,
        validation_data=(test_windows, test_labels),
        batch_size=128,
    )


if __name__ == "__main__":
    main()
