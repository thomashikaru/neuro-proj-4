import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
import logging
import random


class Example:
    def __init__(self, seq):
        self.seq = np.array(seq)

        self.item_vectors = np.zeros((self.seq.size, self.seq.max() + 1))
        self.item_vectors[np.arange(self.seq.size), self.seq] = 1

        self.rank_vectors = np.zeros((self.seq.size, self.seq.size))
        self.rank_vectors[np.arange(self.seq.size), np.arange(self.seq.size)] = 1


class WMModel:
    def __init__(self, N, R, alpha):
        self.N = N
        self.R = R
        self.alpha = alpha
        self.hidden = np.zeros((N * R))
        self.weights = np.zeros((N * R, math.factorial(N)))
        self.output = np.zeros(math.factorial(N))
        self.lookup = [list(x) for x in permutations(range(N))]

    def clear_units(self):
        self.hidden = np.zeros((self.N * self.R))
        self.output = np.zeros(math.factorial(self.N))

    def step(self, item_input, rank_input):
        logging.debug(f"item input shape: {item_input.shape}")
        logging.debug(f"rank input shape: {rank_input.shape}")
        logging.debug(f"rank input shape: {item_input}")
        logging.debug(f"rank input shape: {rank_input}")
        delta_h = np.matmul(item_input.T, rank_input).flatten()
        logging.debug(f"delta_h shape: {delta_h.shape}")
        self.hidden += delta_h
        logging.debug(f"weights shape: {self.weights.shape}")
        a = np.matmul(self.hidden.reshape(1, -1), self.weights)
        logging.debug(f"a shape: {a.shape}")
        self.output = np.exp(a) / np.sum(np.exp(a))
        logging.debug(f"o shape: {self.output.shape}")
        logging.debug(self.output[0, :10])

    def backward(self, target):
        delta_w = self.alpha * np.matmul(
            self.hidden.reshape(-1, 1), (target - self.output)
        )
        logging.debug(f"delta w shape: {delta_w.shape}")
        self.weights += delta_w

    def do_one_trial(self, example, target):
        logging.debug(f"Example: {example.seq}")
        self.clear_units()
        for i in range(example.item_vectors.shape[0]):
            item_vector = example.item_vectors[i, :].reshape(1, -1)
            rank_vector = example.rank_vectors[i, :].reshape(1, -1)
            self.step(item_vector, rank_vector)
        self.backward(target)
        pred = np.argmax(self.output[0])
        logging.debug(f"Prediction: {self.lookup[pred]}")
        return self.lookup[pred]

    def plot_weights(self):
        plt.clf()
        plt.figure(dpi=150)
        p = sns.heatmap(self.weights)
        plt.savefig("img/weights.png")

    def plot_hidden_units(self, seq):
        example = Example(seq)
        self.clear_units()
        for i in range(example.item_vectors.shape[0]):
            item_vector = example.item_vectors[i, :].reshape(1, -1)
            rank_vector = example.rank_vectors[i, :].reshape(1, -1)
            self.step(item_vector, rank_vector)
        plt.figure(dpi=150)
        p = sns.heatmap(self.hidden.reshape((self.N, self.R)))
        plt.savefig("img/hidden_units.png")


def test():
    model = WMModel(6, 6, 0.001)

    example = Example([0, 1, 2, 3, 4, 5])
    target = np.zeros((1, 720))
    target[0, 0] = 1
    pred = model.do_one_trial(example, target)

    example = Example([0, 1, 2, 3, 5, 4])
    target = np.zeros((1, 720))
    target[0, 1] = 1
    pred = model.do_one_trial(example, target)


def test2():

    base_seq = [0, 1, 2, 3, 4, 5]
    S = math.factorial(len(base_seq))

    # indices:
    idxs = np.arange(S)
    random.shuffle(idxs)

    # examples: all permutations of the sequence
    examples = np.array([Example(x) for x in permutations(base_seq)])
    examples = examples[idxs]

    # targets: one-hot vectors of length N!
    targets = np.zeros((S, S))
    targets[np.arange(S), np.arange(S)] = 1
    targets = targets[idxs]

    model = WMModel(6, 6, 0.001)
    n_epochs = 10

    for epoch in range(n_epochs):
        for i in range(len(examples)):
            model.do_one_trial(examples[i], targets[i, :].reshape(1, -1))

    for i in range(len(examples)):
        pred = model.do_one_trial(examples[i], targets[i, :].reshape(1, -1))
        logging.info(f"{examples[i].seq} -> {pred}")

    model.plot_hidden_units([0, 1, 2, 3, 4, 5])
    model.plot_weights()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test2()
