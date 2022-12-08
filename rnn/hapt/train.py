import pickle
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


@dataclass
class LoadedExperiment:
    uid: int
    eid: int
    activities: List[np.ndarray]
    labels: List[int]

    def filter_by_label(self, labels: Sequence[int]) -> "LoadedExperiment":
        return LoadedExperiment(
            self.uid,
            self.eid,
            [x for x, y in zip(self.activities, self.labels) if y in labels],
            [y for y in self.labels if y in labels],
        )


class SimpleRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.project = nn.Linear(input_dim, hidden_dim)
        self.recurrent = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_dim = X.shape
        recurrent_state = torch.zeros(batch_size, self.hidden_dim)
        states = []

        for i in range(seq_len):
            timestep = X[:, i, :]
            timestep_project = F.tanh(self.project(timestep))

            timestep_state = torch.cat([timestep_project, recurrent_state], dim=1)
            recurrent_state = F.tanh(self.recurrent(timestep_state))
            states.append(recurrent_state)

        return torch.stack(states).permute(1, 0, 2)


class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.project = nn.Linear(input_dim, hidden_dim)
        self.forget_recurrent = nn.Linear(hidden_dim * 2, hidden_dim)
        self.input_recurrent = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cell_recurrent = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_recurrent = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_dim = X.shape
        recurrent_state = torch.zeros(batch_size, self.hidden_dim)
        recurrent_cell = torch.zeros(batch_size, self.hidden_dim)
        states = []

        for i in range(seq_len):
            timestep = X[:, i, :]
            timestep_project = F.tanh(self.project(timestep))
            gate_input = torch.cat([timestep_project, recurrent_state], dim=1)

            forget_gate = torch.sigmoid(self.forget_recurrent(gate_input))
            input_gate = torch.sigmoid(self.input_recurrent(gate_input))
            cell_gate = F.tanh(self.cell_recurrent(gate_input))
            output_gate = torch.sigmoid(self.output_recurrent(gate_input))

            recurrent_cell = forget_gate * recurrent_cell + input_gate * cell_gate
            recurrent_state = output_gate * F.tanh(recurrent_cell)
            states.append(recurrent_state)

        return torch.stack(states).permute(1, 0, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(input_dim, output_dim * num_heads)
        self.key = nn.Linear(input_dim, output_dim * num_heads)
        self.value = nn.Linear(input_dim, output_dim * num_heads)
        self.output = nn.Linear(output_dim * num_heads, output_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_dim = X.shape
        X_project = self.project(X)
        X_project = X_project.reshape(batch_size, seq_len, self.num_heads, -1)
        X_project = X_project.transpose(1, 2)

        X_project = X_project / np.sqrt(X_project.shape[-1])
        X_project = F.softmax(X_project, dim=-1)
        X_project = X_project.transpose(1, 2)

        return X_project.reshape(batch_size, seq_len, -1)


class HAPTModel(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.rnn = nn.Sequential(
            LSTM(6, hidden_dim),
            LSTM(hidden_dim, hidden_dim),
        )
        self.classes = nn.Linear(hidden_dim, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        recurrent_state = self.rnn(X)[:, -1]
        return self.classes(recurrent_state)


def load_data() -> List[LoadedExperiment]:
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
    data = [data.filter_by_label({2, 3}) for data in data]
    return data


def print_data_stats(data: List[LoadedExperiment]):
    print(f"Loaded {len(data)} experiments")
    print(f"Total number of activities: {sum(len(x.activities) for x in data)}")
    print()
    for label in {2, 3}:
        label_data = [
            [
                len(exp_activity)
                for (exp_activity, exp_label) in zip(
                    experiment.activities, experiment.labels
                )
                if exp_label == label
            ]
            for experiment in data
        ]
        label_data = sum(label_data, [])
        print(f"Number of activities for class {label}: {len(label_data)}")
        print(f"Activity lengths for class {label}:")
        print(f"Min: {min(label_data)}")
        print(f"Max: {max(label_data)}")
        print(f"Mean: {sum(label_data) / len(label_data)}")
        print()


def split_into_train_test(
    data: List[LoadedExperiment], test_size: float
) -> Tuple[List[LoadedExperiment], List[LoadedExperiment]]:
    split_idx = int(len(data) * (1 - test_size))
    return data[:split_idx], data[split_idx:]


def get_mean_std(data: List[LoadedExperiment]) -> Tuple[np.ndarray, np.ndarray]:
    all_activities = sum([experiment.activities for experiment in data], [])
    all_activities = np.concatenate(all_activities, axis=0)
    print(all_activities.shape)
    return np.mean(all_activities, axis=0, keepdims=True), np.std(
        all_activities, axis=0, keepdims=True
    )


def metrics_for_sequence(
    model: HAPTModel, X: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    y_pred = model(X)
    loss = F.cross_entropy(y_pred, y)
    accuracy = (y_pred.argmax(dim=1) == y).float().mean()
    return loss, accuracy


def train(
    model: HAPTModel,
    train_data: List[LoadedExperiment],
    test_data: List[LoadedExperiment],
    epochs: int,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    standardize: Tuple[np.ndarray, np.ndarray],
):
    mean, std = standardize

    for epoch in range(epochs):
        train_loss = 0
        train_loss_metric = 0
        steps = 0
        for experiment in tqdm(train_data):
            for activity, label in zip(experiment.activities, experiment.labels):
                activity = (activity - mean) / std
                activity_tensor = torch.tensor(activity)[None, :, :].to(torch.float32)
                label_tensor = torch.tensor([label]).to(torch.long) - 2

                loss, _ = metrics_for_sequence(model, activity_tensor, label_tensor)
                train_loss += loss
                train_loss_metric += loss.item()
                steps += 1

                if steps == batch_size:
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                    train_loss = 0
                    steps = 0
        if steps > 0:
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        test_loss = 0
        test_acc = 0
        steps = 0
        for experiment in tqdm(test_data):
            for activity, label in zip(experiment.activities, experiment.labels):
                activity = (activity - mean) / std
                activity_tensor = torch.tensor(activity)[None, :, :].to(torch.float32)
                label_tensor = torch.tensor([label]).to(torch.long) - 2

                loss, acc = metrics_for_sequence(model, activity_tensor, label_tensor)
                test_loss += loss.item()
                test_acc += acc.item()
                steps += 1
        test_loss /= steps
        test_acc /= steps

        print(
            f"Epoch {epoch}: train loss {train_loss:.6f}, test loss {test_loss:.6f}, test accuracy {test_acc:.6f}"
        )


def main():
    data = load_data()
    print_data_stats(data)
    train_data, test_data = split_into_train_test(data, 0.2)
    standardize = get_mean_std(train_data)

    model = HAPTModel(20, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, train_data, test_data, 10, optimizer, 32, standardize)


main()
