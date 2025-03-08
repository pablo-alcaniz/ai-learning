from model import NeuralNetwork
import numpy as np
import pandas as pd
import os


sizes = [20, 10]

nn = NeuralNetwork(sizes)
nn.train_DATA_PATH = "src/data/mnist_train.csv"
nn.activation_functions = [
    "relu",
    "softmax"
]
nn.adam_beta1 = 0.9
nn.adam_beta2 = 0.999
nn.adam_eps = 1E-8
nn.optimizer = "adam"
params = nn.train(epochs=300, lr=0.001, batch_size=None, train_type="complete")
