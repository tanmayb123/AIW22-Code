import random

import numpy as np
import tensorflow as tf

from train import MNISTModel, load_mnist


def load_model():
    model = MNISTModel()
    model.build(input_shape=(None, 28, 28, 1))
    model.load_weights("mnist_model.h5")
    return model


def determine_interesting_channels(w, k):
    argsorted = w.argsort()
    most_positive = argsorted[-k:]
    most_negative = argsorted[:k]
    most_neutral = np.abs(w).argsort()[:k]

    print("Most positively correlated channels:")
    for i in range(k):
        chan = most_positive[i]
        print(f"{chan}: {w[chan]:.4f}")

    print("Most negatively correlated channels:")
    for i in range(k):
        chan = most_negative[i]
        print(f"{chan}: {w[chan]:.4f}")

    print("Least correlated channels:")
    for i in range(k):
        chan = most_neutral[i]
        print(f"{chan}: {w[chan]:.4f}")

    return most_positive, most_negative, most_neutral


def pick_random_for_class(y, cls):
    return random.choice(np.where(y == cls)[0])


def print_conv_score_for_example(x, model, conv_pos, conv_neg):
    pos_idx, pos_weight = conv_pos
    neg_idx, neg_weight = conv_neg
    embeddings = model.embed(x)
    pos_scores = embeddings[:, pos_idx] * pos_weight
    neg_scores = embeddings[:, neg_idx] * neg_weight
    return pos_scores + neg_scores


def main():
    model = load_model()
    w, _ = model.classes.get_weights()
    w = w.T

    chans = []
    for i in range(10):
        print(f"---- Class {i}")
        pos, neg, _ = determine_interesting_channels(w[i], 5)
        print("\n\n")

        chans.append([(pos[0], w[i][pos[0]]), (neg[0], w[i][neg[0]])])

    for i in range(10):
        print(f"Most similar classes to {i}:")
        print(
            np.argsort(
                [
                    np.dot(w[i], w[j]) / (np.linalg.norm(w[i]) * np.linalg.norm(w[j]))
                    for j in range(10)
                ]
            )[::-1][1:5]
        )
        print()

    X_train, y_train, X_test, y_test = load_mnist(False)

    for k in range(10):
        s = []
        for _ in range(300):
            idxs = [pick_random_for_class(y_test, i) for i in range(10)]
            example = X_test[idxs]
            s.append(print_conv_score_for_example(example, model, *chans[k]))
        s = np.array(s)
        s = np.mean(s, axis=0)
        s += np.abs(s.min())
        s /= s.max()
        s /= s.sum()
        for i in range(10):
            print(f"{s[i] * 100:.2f}{'*' if s[i] == s.max() else ''}", end="\t")
        print()


if __name__ == "__main__":
    main()
