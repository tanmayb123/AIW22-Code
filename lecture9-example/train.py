import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import (
    LSTM,
    RNN,
    Activation,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.rnn import LayerNormLSTMCell


def construct_model():
    input_layer = Input((24, 10))

    embed_weather = Dense(32, activation="gelu")(input_layer)
    embed_weather = BatchNormalization()(embed_weather)
    embed_weather = Dropout(0.2)(embed_weather)

    weather_rnn = Bidirectional(
        RNN(LayerNormLSTMCell(32, recurrent_dropout=0.2), return_sequences=True)
    )(embed_weather)
    weather_rnn = Dropout(0.2)(weather_rnn)

    weather_rnn = Bidirectional(
        RNN(LayerNormLSTMCell(32, recurrent_dropout=0.2), return_sequences=False)
    )(weather_rnn)
    weather_rnn = Dropout(0.2)(weather_rnn)

    output_radiation = Dense(1)(weather_rnn)

    model = Model(inputs=input_layer, outputs=output_radiation)

    return model


def construct_contrastive_model(model):
    input_layer1 = Input((24, 10))
    out1 = model(input_layer1)

    input_layer2 = Input((24, 10))
    out2 = model(input_layer2)

    merged = Concatenate()([out1, out2])
    merged = Activation("softmax")(merged)

    contrastive_model = Model(inputs=[input_layer1, input_layer2], outputs=merged)
    contrastive_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(5e-4),
        metrics=["accuracy"],
    )

    return contrastive_model


def load_data_file(filename):
    data = pickle.load(open(filename, "rb"))
    X = [x[0] for x in data]
    y = [x[1] for x in data]
    return np.array(X), np.array(y)


def get_mean_std(X):
    mean = np.mean(X, axis=(0, 1))
    std = np.std(X, axis=(0, 1))
    return mean, std


def standardize(X, mean, std):
    return (X - mean) / std


def create_pairs(X, y, n):
    samples = random.sample(range(len(X)), n * 2)
    paira = []
    pairb = []
    labels = []
    for i in range(0, len(samples), 2):
        paira.append(X[samples[i]])
        pairb.append(X[samples[i + 1]])
        labels.append(1 if y[samples[i]] < y[samples[i + 1]] else 0)
    return np.array(paira), np.array(pairb), np.array(labels)


def train(model, X, y, test_X, test_y):
    test_sets = []
    for _ in range(10):
        test_sets.append(create_pairs(test_X, test_y, len(test_X) // 2))

    for batch in range(1000):
        paira, pairb, labels = create_pairs(X, y, 64)
        print_metrics = batch % 100 == 0
        metrics = model.train_on_batch(
            [paira, pairb], labels, return_dict=True, reset_metrics=print_metrics
        )
        if print_metrics:
            print(
                f'Batch {batch} loss: {metrics["loss"]}, accuracy: {metrics["accuracy"]}'
            )
            test_results = []
            for test_set in test_sets:
                metrics = model.evaluate(
                    [test_set[0], test_set[1]], test_set[2], return_dict=True, verbose=0
                )
                test_results.append(metrics)
            print(
                f'Batch {batch} test loss: {np.mean([x["loss"] for x in test_results])}, accuracy: {np.mean([x["accuracy"] for x in test_results])}'
            )
            print()


def main():
    train_x, train_y = load_data_file("weather_rad_train.pkl")
    test_x, test_y = load_data_file("weather_rad_test.pkl")

    mean, std = get_mean_std(train_x)
    train_x = standardize(train_x, mean, std)
    test_x = standardize(test_x, mean, std)

    model = construct_model()
    contrastive_model = construct_contrastive_model(model)

    train(contrastive_model, train_x, train_y, test_x, test_y)

    model.save("radiation.h5")

    predict_y = model.predict(test_x, batch_size=256, verbose=1)

    plt.scatter(test_y, predict_y)
    plt.savefig("result.png")


if __name__ == "__main__":
    main()
