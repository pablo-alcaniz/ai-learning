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

model = NeuralNetwork(sizes = [128, 60, 50, 10])
model.activation_functions = [
    "relu",
    "tanh",
    "sigmoid", 
    "softmax"
]

data, labels = model.data('src/data/mnist_train.csv')
parametros = model.init_model(data)
parametros = model.forward_prop(parametros, data)
parametros = model.backward_prop(parametros, data, labels)
model.optimizer = "adam"
model.adam_beta1 = 0.9
model.adam_beta2 = 0.999
model.lr = 0.01
model.adam_eps = 1E-8

adam = model.adam_init(parametros)
iter = 1
while iter < 100:
    parametros, adam = model.update_params(parametros, adam, model.lr, model.optimizer, iter)
    print(iter)
    iter += 1