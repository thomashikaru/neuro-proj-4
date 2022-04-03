import logging
import numpy as np
from itertools import permutations
from model import WMModel
import random
import math
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd


sns.set_style("dark")


def test_transpositions(weights_filename):
    logging.basicConfig(level=logging.INFO)

    # load a pre-trained model
    model = WMModel(6, 6, 0.001, delta=0.4, sigma=0.5, nu=0.0)
    model.load_weights(weights_filename)

    # set base sequence
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

    # iterate over test set and get predictions
    transposition_dists = []
    for i in range(len(sequences)):
        result = model.do_one_trial(
            sequences[i], targets[i].reshape(1, -1), compute_metrics=True
        )
        logging.debug(f"{sequences[i]} -> {result['pred']}")
        transposition_dists.append(result["transposition_dists"])

    # aggregate transposition distances
    transposition_dists = np.array(transposition_dists).flatten()
    dist_counts = Counter(transposition_dists).most_common()
    logging.info(f"Transposition Distances: {dist_counts}")

    positions, counts = zip(*dist_counts)
    s = sum(counts)
    pcts = [x / s for x in counts]

    df = pd.DataFrame(
        {
            "distance": positions,
            "transposition_prob": pcts,
            "training_epochs": int(weights_filename.split("_")[1]),
        }
    )

    return df


if __name__ == "__main__":

    epochs = [50] + list(range(250, 2501, 250))
    dfs = []

    for e in epochs:
        dfs.append(test_transpositions(f"weights/weights_{e}_epochs.npy"))

    df = pd.concat(dfs)

    # create figure
    plt.clf()
    plt.figure(dpi=150)
    p = sns.lineplot(
        data=df,
        x="distance",
        y="transposition_prob",
        hue="training_epochs",
        legend="full",
        marker="o",
        palette="crest",
    )
    p.set_title("Transposition Gradient Effect", fontsize=16)
    p.set_xlabel("Transposition Distance", fontsize=14)
    p.set_ylabel("Transposition Probability", fontsize=14)
    p.set_xticks(range(6))
    plt.savefig("img/transposition_dists")
