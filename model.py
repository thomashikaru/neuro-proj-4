import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns


class WMModel:
    def __init__(self, N, R, alpha):
        self.N = N
        self.R = R
        self.alpha = alpha
        self.hidden = np.zeros((N * R))
        self.weights = np.zeros((N * R, math.factorial(N)))
        self.output = np.zeros(math.factorial(N))

    def step(self, item_input, rank_input, target):
        print("item input shape:", item_input.shape)
        print("rank input shape:", rank_input.shape)
        delta_h = np.matmul(item_input.T, rank_input).flatten()
        print("delta_h shape:", delta_h.shape)
        self.hidden += delta_h
        print("weights shape:", self.weights.shape)
        a = np.matmul(self.hidden.reshape(1, -1), self.weights)
        print("a shape:", a.shape)
        o = np.exp(a) / np.sum(np.exp(a))
        print("o shape:", o.shape)
        print(o)

        delta_w = self.alpha * np.matmul(self.hidden.reshape(-1, 1), (target - o))
        print("delta w shape:", delta_w.shape)

        self.weights += delta_w
        return o

    def train(self, item_inputs, rank_inputs, targets):
        for i, r, t in zip(item_inputs, rank_inputs, targets):
            self.step(i, r, t)


def test():
    model = WMModel(6, 6, 0.001)
    target = np.zeros((1, 720))
    target[0, 0] = 1
    for i in range(2):
        model.step(
            np.array([[1, 2, 3, 4, 5, 6]]), np.array([[1, 2, 3, 4, 5, 6]]), target,
        )


if __name__ == "__main__":
    test()
