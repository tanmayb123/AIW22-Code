import numpy as np

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

from tensorflow.keras.models import load_model

def load_pretrained_model():
    loaded_model = load_model('resnet50v2_classifier.h5')
    return loaded_model

def load_test_data_and_labels():
    images = np.load("test_images.npy")
    labels = np.load("test_labels.npy")
    return images, labels

def main():
    model = load_pretrained_model()
    test_images, test_labels = load_test_data_and_labels()
    predictions = model.predict(test_images, verbose=1, batch_size=256)
    predicted_classes = predictions.argmax(axis=1)

    accuracy = (predicted_classes == test_labels).astype(np.int32).sum() / len(test_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    matrix = np.zeros((120, 120))
    for i in range(len(test_labels)):
        matrix[test_labels[i], predicted_classes[i]] += 1
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    matrix[np.where(matrix.sum(axis=1) == 0)] = 1

    for i in range(120):
        confused_by = matrix[i].argsort()[::-1][:5]
        confusion_scores = matrix[i][confused_by]
        print(f"Class {i} is confused by:")
        for j in range(5):
            print(f"\tClass {confused_by[j]} with {confusion_scores[j] * 100:.2f}%")
        print()

if __name__ == "__main__":
    main()
