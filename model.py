import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
import logging
import random
from tqdm import tqdm


def transposition_dists(a, b):
    """return the a vector of the same length as a and b where the value of the
    i-th element is the absolute value of the transposition distance between the
    i-th element of a and its corresponding value in b

    Args:
        a (object): numpy array
        b (object): numpy array

    Returns:
        object: numpy array of transposition distances
    """
    c = np.array([np.where(a == i) for i in b]).flatten()
    d = np.arange(len(c))
    return np.abs(c - d)


class WMModel:
    """
    Working Memory Model based on Botvinick and Watanabe
    """

    class Example:
        def __init__(self, seq):
            self.seq = np.array(seq)

    def __init__(self, N, R, alpha=0.001, delta=0.5, sigma=0.5, nu=0.0):
        """create instance of a working memory model

        Args:
            N (int): number of unique items that can be represented
            R (int): number of ranks that can be represented
            alpha (float, optional): learning rate. Defaults to 0.001.
            delta (float, optional): dissimilarity between high-responsive items and low-responsive items. Defaults to 0.5.
            sigma (float, optional): scaling constant. Defaults to 0.5.
            nu (float, optional): random noise variance. Defaults to 0.0.
        """
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

        # example.item_vectors = np.full((example.seq.size, example.seq.max() + 1), 0)
        # example.item_vectors[np.arange(example.seq.size), example.seq] = 1
        # example.rank_vectors = np.zeros((example.seq.size, example.seq.size))
        # example.rank_vectors[np.arange(example.seq.size), np.arange(example.seq.size)] = 1

        example.item_vectors = np.full(
            (example.seq.size, example.seq.max() + 1), 1 - self.delta
        )
        example.item_vectors[np.arange(example.seq.size), example.seq] = 1

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

    def do_one_trial(self, seq, target, compute_metrics=False):
        example = self.seq_to_example(seq)
        logging.debug(f"Example: {example.seq}")

        self.clear_units()
        for i in range(example.item_vectors.shape[0]):
            item_vector = example.item_vectors[i].reshape(1, -1)
            rank_vector = example.rank_vectors[i].reshape(1, -1)
            self.step(item_vector, rank_vector)
        self.backward(target)

        result = {}
        pred_idx = np.argmax(self.output[0])
        logging.debug(f"Prediction: {self.lookup[pred_idx, :]}")
        result["pred_idx"] = pred_idx
        result["pred"] = self.lookup[pred_idx, :]

        if compute_metrics:
            result["pos_acc"] = example.seq == result["pred"]
            result["transposition_dists"] = transposition_dists(
                example.seq, result["pred"]
            )
        return result

    def plot_weights(self, filename="img/weights.png"):
        plt.clf()
        plt.figure(figsize=(12, 5), dpi=150)
        p = sns.heatmap(self.weights)
        plt.savefig(filename)

    def plot_hidden_units(self, seq, filename="img/hidden_units.png"):
        example = self.seq_to_example(seq)
        self.clear_units()
        for i in range(example.item_vectors.shape[0]):
            item_vector = example.item_vectors[i].reshape(1, -1)
            rank_vector = example.rank_vectors[i].reshape(1, -1)
            self.step(item_vector, rank_vector)
        plt.clf()
        plt.figure(dpi=150)
        p = sns.heatmap(self.hidden.reshape((self.N, self.R)))
        plt.savefig(filename)

    def save_weights(self, weights_file):
        np.save(weights_file, self.weights)

    def load_weights(self, weights_file):
        self.weights = np.load(weights_file)


def test_basic():
    # test basic forward and backward functions

    logging.basicConfig(level=logging.DEBUG)
    model = WMModel(6, 6, 0.001, delta=0.5, sigma=0.5, nu=0.09)

    seq = [0, 1, 2, 3, 4, 5]
    target = np.zeros((1, 720))
    target[0, 0] = 1
    result = model.do_one_trial(seq, target)

    seq = [0, 1, 2, 3, 5, 4]
    target = np.zeros((1, 720))
    target[0, 1] = 1
    result = model.do_one_trial(seq, target)


def test_item_rank_vectors():
    # test that item and rank vectors are computed correctly
    logging.basicConfig(level=logging.DEBUG)
    model = WMModel(6, 6, 0.001, delta=0.5, sigma=0.5, nu=0.0)
    seq = [0, 1, 2, 3, 4, 5]
    e = model.seq_to_example(seq)
    logging.info(e.item_vectors)
    logging.info(e.rank_vectors)


def plot_model(weights_file):
    # test plotting of hidden units and weights
    logging.basicConfig(level=logging.INFO)
    model = WMModel(6, 6)
    seq = [0, 1, 2, 3, 4, 5]
    model.load_weights(weights_file)
    model.plot_hidden_units(seq)
    model.plot_weights()


def train_model(n_epochs):
    logging.basicConfig(level=logging.INFO)
    model = WMModel(6, 6, 0.001, delta=0.4, sigma=0.5, nu=0.0)

    # base sequence, permutations, indices, and targets (one-hot vectors of length N!)
    base_seq = [0, 1, 2, 3, 4, 5]
    S = math.factorial(len(base_seq))
    ordered_sequences = np.array([list(x) for x in permutations(base_seq)])
    idxs = np.arange(S)
    ordered_targets = np.zeros((S, S))
    ordered_targets[np.arange(S), np.arange(S)] = 1

    for _ in tqdm(range(n_epochs)):

        # shuffle the data
        random.shuffle(idxs)
        sequences = ordered_sequences[idxs]
        targets = ordered_targets[idxs]

        for i in range(len(sequences)):
            model.do_one_trial(sequences[i], targets[i].reshape(1, -1))

    pos_accs = []
    transposition_dists = []
    for i in range(len(ordered_sequences)):
        result = model.do_one_trial(
            ordered_sequences[i],
            ordered_targets[i].reshape(1, -1),
            compute_metrics=True,
        )
        logging.info(f"{ordered_sequences[i]} -> {result['pred']}")
        pos_accs.append(result["pos_acc"])
        transposition_dists.append(result["transposition_dists"])

    # model.nu = 0.09
    model.plot_hidden_units([0, 1, 2, 3, 4, 5], f"img/hidden_units_012345.png")
    model.plot_weights(f"img/weights_{n_epochs}.png")
    model.save_weights(f"weights/weights_{n_epochs}_epochs.npy")

    pos_accs = np.array(pos_accs).mean(axis=0)
    logging.info(f"Positional Accuracies: {pos_accs}")


if __name__ == "__main__":
    train_model(n_epochs=250)
