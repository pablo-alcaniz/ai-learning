from model import NeuralNetwork
import numpy as np
import pandas as pd
import os


sizes = [20, 10]

nn = NeuralNetwork(sizes)

nn.train_DATA_PATH = "src/data/mnist_train.csv"
nn.test_DATA_PATH = "src/data/mnist_test.csv"

nn.activation_functions = [
    "relu",
    "sigmoid"
]
nn.adam_beta1 = 0.9
nn.adam_beta2 = 0.99
nn.adam_eps = 1E-8
nn.optimizer = "adam"
nn.batch_prints = 5
params = nn.train(epochs=1, lr=0.01, batch_size=128, train_type="batch", log = True)

nn.SAVE_DIR = "src/model"
nn.SAVE_NAME = "model-1"

nn.save_model(params)





