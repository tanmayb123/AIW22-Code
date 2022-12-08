import tensorflow as tf
from tqdm import tqdm

REF_1 = tf.constant([2, -2], dtype=tf.float32)
REF_2 = tf.constant([10, 8], dtype=tf.float32)
REF_3 = tf.constant([-1, 6], dtype=tf.float32)

REF_1_DIST = tf.constant(4, dtype=tf.float32)
REF_2_DIST = tf.constant(10, dtype=tf.float32)
REF_3_DIST = tf.constant(5, dtype=tf.float32)


def error(predicted_coordinate):
    """
    Calculate the error between the distances from the predicted coordinate
    to the references and the actual distances.
    """
    ref_1_dist = tf.norm(predicted_coordinate - REF_1)
    ref_2_dist = tf.norm(predicted_coordinate - REF_2)
    ref_3_dist = tf.norm(predicted_coordinate - REF_3)

    return (
        tf.abs(REF_1_DIST - ref_1_dist)
        + tf.abs(REF_2_DIST - ref_2_dist)
        + tf.abs(REF_3_DIST - ref_3_dist)
    )


def optimize():
    """
    Optimize the error function to find the predicted coordinate.
    """
    predicted_coordinate = tf.Variable([0, 0], dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    pbar = tqdm(range(1000))
    for _ in pbar:
        optimizer.minimize(
            lambda: error(predicted_coordinate), var_list=[predicted_coordinate]
        )
        loss = error(predicted_coordinate)
        pbar.set_description(f"Loss: {loss.numpy()}")

    return predicted_coordinate


def main():
    """
    Main function.
    """
    predicted_coordinate = optimize()
    print(predicted_coordinate)


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()
