import numpy as np
import pandas as pd

class NeuralNetwork():

    activation_functions = []       #Vector with the activation function desired for each layer in sizes: activation_function = ['relu', 'relu', 'sigmoid', ...]
    epochs: int                     #epochs for the training
    lr: float                       #Learning rate

    def __init__(self, sizes):
        self.sizes = sizes      #sizes of the layers including outputs but excluding inputs               

    def data(self, train_DATA_PATH, test_DATA_PATH=None):

        train_data_raw = pd.read_csv(train_DATA_PATH)               #read csv file with pandas
        train_data_raw = train_data_raw.fillna(0)                   #prevent NaN
        train_data_raw = np.array(train_data_raw)                   #convert pandas type to a numpy one
        np.random.shuffle(train_data_raw)                           #shuffle data to prevent overfitting
        train_data = train_data_raw[:,1:]                           #split the train data from the labels
        train_data = np.transpose(train_data/train_data.max())      
        train_labels = train_data_raw[:,0]                          #take the labels for the training

        if test_DATA_PATH is None:      
            return train_data, train_labels                         #if user doesn't specifies a test set return the train data 
        else:
            test_data_raw = pd.read_csv(test_DATA_PATH)
            test_data_raw = test_data_raw.fillna(0)
            test_data_raw = np.array(test_data_raw) 
            test_data = test_data_raw[:,1:]
            test_data = np.transpose(test_data)
            test_labels = test_data_raw[:,0]
            return train_data, train_labels, test_data, test_labels
    
    def init_model(self, train_data):
        train_layer_dim, train_examples = train_data.shape                                          #shape of the train data tensor 
        model_params = {}                                                                           #init the dict where the parameters of the network will be stored
        for i in range(len(self.sizes)):                                                            #loop to define and random initialization of the parameters
            model_params["b_"+str(i)] = np.random.rand(self.sizes[i],train_examples) - 0.5          #bias init
            if i == 1:
                model_params["W_"+str(i)] = np.random.rand(self.sizes[i],train_layer_dim) - 0.5     #the first layer always depend on the dimension of the input data
            else:
                model_params["W_"+str(i)] = np.random.rand(self.sizes[i],self.sizes[i-1]) - 0.5     #rest of the layers
        return model_params

    def activation_func(Z, func, derivative=None):                                                  #a function that contains activation functions and their derivatives
        if func == "relu":                                                                          #to activate the derivative functionallity activation_func(Z, "relu", derivative=True)
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
            if derivative is None:
                return np.exp(Z)/sum(np.exp(Z))
            if derivative is True:
                raise Exception("Softmax derivative its not implemented yet. Please use another.")
        else:
            raise Exception("Activation function not recognized")

    def delta(self,i,j):        #kronecker delta
        if i == j: 
            return 1
        else:
            return 0
        
    def forward_prop(self, model_params):
        self.activation_functions