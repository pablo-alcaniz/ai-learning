from model import NeuralNetwork
import numpy as np
import pandas as pd
import os
import random

sizes = [ 250, 10, 250, 784]

nn = NeuralNetwork(sizes)

nn.latent_index = 2

nn.train_DATA_PATH = "src/data/mnist_train.csv"
nn.test_DATA_PATH = "src/data/mnist_test.csv"

nn.activation_functions = [
    "sigmoid",
    "sigmoid",
    "sigmoid",
    "sigmoid"
]

nn.adam_beta1 = 0.9
nn.adam_beta2 = 0.999
nn.adam_eps = 1E-8
nn.optimizer = "adam"
nn.batch_prints = 5
params = nn.train(epochs=2, lr=0.001, batch_size=64, train_type="batch", log = True)

nn.SAVE_DIR = "model"
nn.SAVE_NAME = "model-1"
nn.save_model(params)

nn.latent_pred(random.randint(0,9), params)




