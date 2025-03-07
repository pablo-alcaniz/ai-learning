from model import NeuralNetwork
import numpy as np
import pandas as pd
import os


sizes = [10, 10]

nn = NeuralNetwork(sizes)
nn.train_DATA_PATH = "src/data/mnist_train.csv"
nn.activation_functions = [
    "relu",
    "softmax"
]
nn.optimizer = "gd"
params = nn.train(epochs=5, lr=0.5, batch_size=None, train_type="complete")
