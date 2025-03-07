import numpy as np
import pandas as pd

class NeuralNetwork():

    act_func_list = []
    def __init__(self, sizes, epochs, lr):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr

    def data(self, train_DATA_PATH, test_DATA_PATH=None):

        train_data_raw = pd.read_csv(train_DATA_PATH)
        train_data_raw = train_data_raw.fillna(0)
        train_data_raw = np.array(train_data_raw)
        np.random.shuffle(train_data_raw)
        train_data = train_data_raw[:,1:]
        train_data = np.transpose(train_data/train_data.max())
        train_labels = train_data_raw[:,0]

        if test_DATA_PATH is None:
            return train_data, train_labels
        else:
            test_data_raw = pd.read_csv(test_DATA_PATH)
            test_data_raw = test_data_raw.fillna(0)
            test_data_raw = np.array(test_data_raw) 
            test_data = test_data_raw[:,1:]
            test_data = np.transpose(test_data)
            test_labels = test_data_raw[:,0]
            return train_data, train_labels, test_data, test_labels
    
    def init_model(self, train_data):
        train_layer_dim, train_examples = train_data.shape 
        model_params = {}
        for i in range(len(self.sizes)):
            model_params["b_"+str(i)] = np.random.rand(self.sizes[i],train_examples) - 0.5
            if i == 1:
                model_params["W_"+str(i)] = np.random.rand(self.sizes[i],train_layer_dim) - 0.5
            else:
                model_params["W_"+str(i)] = np.random.rand(self.sizes[i],self.sizes[i-1]) - 0.5
        return model_params

    def activation_func(Z, func, derivative=None):
        if func == "relu":
            if derivative is None:
                return max(0,Z)
            if derivative is True:
                return 1 if Z > 0 else 0
        if func == "sigmoid":
            if derivative is None:
                return 1.0 / (1 + np.exp(-Z))
            if derivative is True:
                return 1.0 / (1 + np.exp(-Z)) * (1-1.0 / (1 + np.exp(-Z)))
        if func == "tanh":
            if derivative is None:
                return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
            if derivative is True:
                return 1 - np.power((np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z)), 2)
        if func == "softmax":
            return np.exp(Z)/sum(np.exp(Z))
        else:
            raise Exception("Activation function not recognized")

    def delta(self,i,j):
        if i == j: 
            return 1
        else:
            return 0
        
    def forward_prop(self, model_params):
        self.act_func_list