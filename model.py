import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
import logging
import random


class WMModel:
    """
    Working Memory Model based on Botvinick and Watanabe
    """

    class Example:
        def __init__(self, seq):
            self.seq = np.array(seq)

    def __init__(self, N, R, alpha, delta, sigma, nu):
        self.N = N
        self.R = R
        self.alpha = alpha
        self.delta = delta
        self.sigma = sigma
        self.nu = nu
        self.hidden = np.zeros((N * R))
        self.weights = np.zeros((N * R, math.factorial(N)))
        self.output = np.zeros(math.factorial(N))
        self.lookup = np.array([list(x) for x in permutations(range(N))])

    def seq_to_example(self, seq):
        example = self.Example(seq)

        example.item_vectors = np.full(
            (example.seq.size, example.seq.max() + 1), 1 - self.delta
        )
        example.item_vectors[np.arange(example.seq.size), example.seq] = 1

        # self.rank_vectors = np.zeros((self.seq.size, self.seq.size))
        # self.rank_vectors[np.arange(self.seq.size), np.arange(self.seq.size)] = 1

        example.rank_vectors = []
        for i in range(example.seq.size):
            r = np.full(example.seq.size, i + 1)
            rho = np.arange(example.seq.size) + 1
            a = np.exp(-np.power(np.log(r) - np.log(rho), 2) / (2 * self.sigma ** 2))
            example.rank_vectors.append(a)
        example.rank_vectors = np.array(example.rank_vectors)

        return example

    def clear_units(self):
        self.hidden = np.zeros((self.N * self.R))
        self.output = np.zeros(math.factorial(self.N))

    def step(self, item_input, rank_input):
        logging.debug(f"item input shape: {item_input.shape}")
        logging.debug(f"rank input shape: {rank_input.shape}")
        logging.debug(f"rank input shape: {item_input}")
        logging.debug(f"rank input shape: {rank_input}")
        item_input *= np.random.normal(loc=1, scale=self.nu, size=item_input.shape)
        rank_input *= np.random.normal(loc=1, scale=self.nu, size=rank_input.shape)

        delta_h = np.matmul(item_input.T, rank_input).flatten()
        logging.debug(f"delta_h shape: {delta_h.shape}")
        logging.debug(f"hidden shape: {self.hidden.shape}")
        self.hidden += delta_h
        self.hidden *= np.random.normal(loc=1, scale=self.nu, size=self.hidden.shape)

        logging.debug(f"weights shape: {self.weights.shape}")
        a = np.matmul(self.hidden.reshape(1, -1), self.weights)
        logging.debug(f"a shape: {a.shape}")
        self.output = np.exp(a) / np.sum(np.exp(a))
        logging.debug(f"output shape: {self.output.shape}")
        logging.debug(self.output[0, :10])

    def backward(self, target):
        delta_w = self.alpha * np.matmul(
            self.hidden.reshape(-1, 1), (target - self.output)
        )
        logging.debug(f"delta w shape: {delta_w.shape}")
        self.weights += delta_w

    def do_one_trial(self, seq, target):
        example = self.seq_to_example(seq)
        logging.debug(f"Example: {example.seq}")

        self.clear_units()
        for i in range(example.item_vectors.shape[0]):
            item_vector = example.item_vectors[i, :].reshape(1, -1)
            rank_vector = example.rank_vectors[i, :].reshape(1, -1)
            self.step(item_vector, rank_vector)
        self.backward(target)

        result = {}
        pred_idx = np.argmax(self.output[0])
        logging.debug(f"Prediction: {self.lookup[pred_idx, :]}")
        result["pred_idx"] = pred_idx
        result["pred"] = self.lookup[pred_idx, :]
        result["pos_acc"] = example.seq == result["pred"]
        return result

    def plot_weights(self):
        plt.clf()
        plt.figure(dpi=150)
        p = sns.heatmap(self.weights)
        plt.savefig("img/weights.png")

    def plot_hidden_units(self, seq):
        example = self.seq_to_example(seq)
        self.clear_units()
        for i in range(example.item_vectors.shape[0]):
            item_vector = example.item_vectors[i, :].reshape(1, -1)
            rank_vector = example.rank_vectors[i, :].reshape(1, -1)
            self.step(item_vector, rank_vector)
        plt.clf()
        plt.figure(dpi=150)
        p = sns.heatmap(self.hidden.reshape((self.N, self.R)))
        plt.savefig("img/hidden_units.png")


def test():
    model = WMModel(6, 6, 0.001, delta=0.5, sigma=0.5, nu=0.09)

    seq = [0, 1, 2, 3, 4, 5]
    target = np.zeros((1, 720))
    target[0, 0] = 1
    result = model.do_one_trial(seq, target)

    seq = [0, 1, 2, 3, 5, 4]
    target = np.zeros((1, 720))
    target[0, 1] = 1
    result = model.do_one_trial(seq, target)


def test2():

    model = WMModel(6, 6, 0.001, delta=0.9, sigma=0.5, nu=0.0)

    base_seq = [0, 1, 2, 3, 4, 5]
    S = math.factorial(len(base_seq))

    # indices: for shuffling the input
    idxs = np.arange(S)
    random.shuffle(idxs)

    # sequences: all permutations of the sequence
    sequences = np.array([list(x) for x in permutations(base_seq)])
    sequences = sequences[idxs]

    # targets: one-hot vectors of length N!
    targets = np.zeros((S, S))
    targets[np.arange(S), np.arange(S)] = 1
    targets = targets[idxs]

    n_epochs = 100

    for epoch in range(n_epochs):
        for i in range(len(sequences)):
            model.do_one_trial(sequences[i], targets[i, :].reshape(1, -1))

    pos_accs = []
    for i in range(len(sequences)):
        result = model.do_one_trial(sequences[i], targets[i, :].reshape(1, -1))
        logging.info(f"{sequences[i]} -> {result['pred']}")
        pos_accs.append(result["pos_acc"])

    # model.nu = 0.09
    model.plot_hidden_units([3, 4, 5, 0, 1, 2])
    model.plot_weights()

    pos_accs = np.array(pos_accs).mean(axis=0)
    logging.info(f"Positional Accuracies: {pos_accs}")


def test3():
    model = WMModel(6, 6, 0.001, delta=0.5, sigma=0.5, nu=0.0)
    seq = [0, 1, 2, 3, 4, 5]
    e = model.seq_to_example(seq)
    logging.info(e.item_vectors)
    logging.info(e.rank_vectors)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test2()
