from model import NeuralNetwork
import numpy as np
import pandas as pd
import os



#train_data_raw = pd.read_csv('src/data/mnist_train.csv')
#train_data_raw = train_data_raw.fillna(0)
#train_data_raw = np.array(train_data_raw)
#np.random.shuffle(train_data_raw)
#train_data = train_data_raw[:,1:]
#train_data = np.transpose(train_data/train_data.max())
#train_labels = train_data_raw[:,0]

model = NeuralNetwork(sizes = [128, 60, 50, 128])
model.activation_functions = [
    "relu",
    "tanh",
    "sigmoid", 
    "softmax"
]

data, _ = model.data('src/data/mnist_train.csv')
parametros = model.init_model(data)
parametros = model.forward_prop(parametros, data)