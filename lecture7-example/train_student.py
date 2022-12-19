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
    probabilities_layer = Dense(120)(probabilities_layer)
    model = Model(inputs=model.input, outputs=probabilities_layer, name="Student")

    #model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["accuracy"])
    #model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

class Distiller(Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(preprocess_input(tf.cast(x, tf.float32)), training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

def main():
    (train_images, train_labels), (test_images, test_labels) = load_data(use_existing_shuffle=True, preprocessing=False)

    print("Loaded data")
    print("Train images:", train_images.shape)
    print("Train labels:", train_labels.shape)
    print("Test images:", test_images.shape)
    print("Test labels:", test_labels.shape)

    model = load_student_model()
    teacher = load_pretrained_model()

    distiller = Distiller(student=model, teacher=teacher)
    distiller.compile(
        optimizer=Adam(1e-3),
        metrics=["accuracy"],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    distiller.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels), shuffle=True)

    model.save("mobilenet_classifier.h5")

if __name__ == "__main__":
    main()
