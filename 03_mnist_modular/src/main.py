from model import NeuralNetwork
import numpy as np
import pandas as pd
import os

model = NeuralNetwork(sizes = [128, 64, 10, 10], lr = 0.1, epochs=10)

a = str("A")
model_params =  {
    a: 2+2,
    "b": 4
}

dict = {}
for i in range(10):
    dict["Test_"+str(i)] = np.random.rand(4,4)


#print(dict)

a = [128, 64, 10, 10]
print(len(a))


train_data_raw = pd.read_csv('src/data/mnist_train.csv')
train_data_raw = train_data_raw.fillna(0)
train_data_raw = np.array(train_data_raw)
np.random.shuffle(train_data_raw)
train_data = train_data_raw[:,1:]
train_data = np.transpose(train_data/train_data.max())
train_labels = train_data_raw[:,0]

_,b = train_data.shape

print(train_data.shape)
print(b)

model.act_func_list = [
    "relu",
    "relu",
    "sigmoid"
]

