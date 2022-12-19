import pickle
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

ACTIVITIES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
    7: "STAND_TO_SIT",
    8: "SIT_TO_STAND",
    9: "SIT_TO_LIE",
    10: "LIE_TO_SIT",
    11: "STAND_TO_LIE",
    12: "LIE_TO_STAND",
}
CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

SAMPLING_RATE = 50
WINDOW_SIZE = 50
STRIDE_SIZE = 25


@dataclass
class LoadedExperiment:
    uid: int
    eid: int
    activities: List[np.ndarray]
    labels: List[int]

    def filter_by_labels(self, allowed_labels: Sequence[int]) -> "LoadedExperiment":
        return LoadedExperiment(
            self.uid,
            self.eid,
            [
                activity
                for activity, label in zip(self.activities, self.labels)
                if label in allowed_labels
            ],
            [
                allowed_labels.index(label)
                for label in self.labels
                if label in allowed_labels
            ],
        )

    def get_windows(
        self, window_size: int, stride_size: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        windows = []
        labels = []
        for activity, label in zip(self.activities, self.labels):
            for i in range(0, len(activity) - window_size, stride_size):
                windows.append(activity[i : i + window_size])
                labels.append(label)
        if len(windows) == 0:
            return None
        return np.array(windows), np.array(labels)


def load_data(allowed_labels: Sequence[int]) -> List["LoadedExperiment"]:
    with open("data.pkl", "rb") as datafile:
        data = pickle.load(datafile)
    return [exp.filter_by_labels(allowed_labels) for exp in data]


def split_into_train_test(
    data: List[LoadedExperiment], test_percentage: float
) -> Tuple[List[LoadedExperiment], List[LoadedExperiment]]:
    test_count = int(len(data) * test_percentage)
    return data[test_count:], data[:test_count]


def get_windows_for_experiments(
    experiments: List[LoadedExperiment],
) -> Tuple[np.ndarray, np.ndarray]:
    windows = []
    labels = []
    for exp in experiments:
        exp_data = exp.get_windows(WINDOW_SIZE, STRIDE_SIZE)
        if exp_data is None:
            continue
        exp_windows, exp_labels = exp_data
        windows.append(exp_windows)
        labels.append(exp_labels)
    npwindows = np.concatenate(windows)
    nplabels = np.concatenate(labels)
    return npwindows, nplabels


def load_windowed_dataset() -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    experiments = load_data(CLASSES)
    train_exp, test_exp = split_into_train_test(experiments, 0.2)
    train_windows, train_labels = get_windows_for_experiments(train_exp)
    test_windows, test_labels = get_windows_for_experiments(test_exp)

    print(f"Train windows {train_windows.shape}, labels {train_labels.shape}")
    print(f"Test windows {test_windows.shape}, labels {test_labels.shape}")

    for classidx in range(len(CLASSES)):
        class_train_count = (train_labels == classidx).astype(np.int64).sum()
        class_test_count = (test_labels == classidx).astype(np.int64).sum()
        print(f"Class {classidx} train {class_train_count}, test {class_test_count}")

    return (train_windows, train_labels), (test_windows, test_labels)


if __name__ == "__main__":
    _ = load_windowed_dataset()
