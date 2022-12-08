import glob
import multiprocessing as mp
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import numpy as np
from tqdm import tqdm


@dataclass
class Activity:
    start: int
    end: int
    label: int


@dataclass
class Experiment:
    eid: int
    activities: List[Activity]


@dataclass
class User:
    uid: int
    experiments: List[Experiment]


@dataclass
class LoadedExperiment:
    uid: int
    eid: int
    activities: List[np.ndarray]
    labels: List[int]


def get_full_data() -> List[User]:
    """
    Returns a list of users, each with a list of experiments, and each with their activities.
    """
    labels = np.loadtxt("labels.txt", delimiter=" ")
    raw_data_files = glob.glob("RawData/*.txt")
    users = defaultdict(lambda: User(0, []))
    experiments = defaultdict(lambda: Experiment(0, []))
    for file in raw_data_files:
        match = re.match(r"RawData/([a-z]+)_exp(\d+)_user(\d+).txt", file)
        if match:
            _, experiment_id, user_id = match.groups()
            experiment_id = int(experiment_id)
            user_id = int(user_id)
            users[user_id].uid = user_id
            experiments[experiment_id].eid = experiment_id
            experiments[experiment_id].activities = []
            users[user_id].experiments.append(experiments[experiment_id])
    for label in labels:
        experiment_id, user_id, activity_id, start, end = label
        experiment_id = int(experiment_id)
        user_id = int(user_id)
        activity_id = int(activity_id)
        start = int(start)
        end = int(end)
        experiments[experiment_id].activities.append(Activity(start, end, activity_id))
    return list(users.values())


def load_user_experiments(user: User) -> List[LoadedExperiment]:
    """
    Loads the experiments for a given user.
    """
    experiments = []
    for experiment in user.experiments:
        activities = []
        labels = []
        for activity in experiment.activities:
            acc_data = np.loadtxt(
                f"RawData/acc_exp{experiment.eid:02d}_user{user.uid:02d}.txt",
                delimiter=" ",
            )
            gyro_data = np.loadtxt(
                f"RawData/gyro_exp{experiment.eid:02d}_user{user.uid:02d}.txt",
                delimiter=" ",
            )
            activities.append(
                np.concatenate(
                    [
                        acc_data[activity.start : activity.end],
                        gyro_data[activity.start : activity.end],
                    ],
                    axis=1,
                )
            )
            labels.append(activity.label)
        experiments.append(
            LoadedExperiment(user.uid, experiment.eid, activities, labels)
        )
    return experiments


if __name__ == "__main__":
    data = get_full_data()

    with mp.Pool() as pool:
        loaded = list(tqdm(pool.imap(load_user_experiments, data), total=len(data)))
    loaded = sum(loaded, [])

    with open("data.pkl", "wb") as f:
        pickle.dump(loaded, f)
