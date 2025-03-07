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
            print("INFO: Data loaded correctly")
            print("WARNING: Only train data loaded")   
            return train_data, train_labels                         #if user doesn't specifies a test set return the train data 
        else:
            test_data_raw = pd.read_csv(test_DATA_PATH)
            test_data_raw = test_data_raw.fillna(0)
            test_data_raw = np.array(test_data_raw) 
            test_data = test_data_raw[:,1:]
            test_data = np.transpose(test_data)
            test_labels = test_data_raw[:,0]
            print("INFO: Data loaded correctly")
            print("INFO: Train and test data loaded")    
            return train_data, train_labels, test_data, test_labels
    
    def init_model(self, train_data):
        train_layer_dim, train_examples = train_data.shape                                          #shape of the train data tensor 
        model_params = {}                                                                           #init the dict where the parameters of the network will be stored
        for i in range(len(self.sizes)):                                                            #loop to define and random initialization of the parameters
            model_params["b_"+str(i+1)] = np.random.rand(self.sizes[i],train_examples) - 0.5        #bias init
            if i+1 == 1:
                model_params["W_"+str(i+1)] = np.random.rand(self.sizes[i],train_layer_dim) - 0.5   #the first layer always depend on the dimension of the input data
            else:
                model_params["W_"+str(i+1)] = np.random.rand(self.sizes[i],self.sizes[i-1]) - 0.5   #rest of the layers
        print('INFO: Model parameters init correct')
        return model_params

    def activation_func(self, Z, func, derivative=False):                                           #a function that contains activation functions and their derivatives
        if func == "relu":                                                                          #to activate the derivative functionallity activation_func(Z, "relu", derivative=True)
            if derivative is False:
                return np.maximum(0,Z)
            if derivative is True:
                return Z > 0
        if func == "sigmoid":
            if derivative is False:
                return 1.0 / (1 + np.exp(-Z))
            if derivative is True:
                return 1.0 / (1 + np.exp(-Z)) * (1 - 1.0 / (1 + np.exp(-Z)))
        if func == "tanh":
            if derivative is False:
                return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
            if derivative is True:
                return 1 - np.power((np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z)), 2)
        if func == "softmax":
            if derivative is False:
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
        
    def forward_prop(self, model_params,train_data):
        for i in range(len(self.sizes)):
            if i+1 == 1:
                model_params["Z_"+str(i+1)] = model_params["W_"+str(i+1)].dot(train_data) + model_params["b_"+str(i+1)]
                model_params["A_"+str(i+1)] = self.activation_func(model_params["Z_"+str(i+1)], self.activation_functions[i], derivative=False)
            else:
                model_params["Z_"+str(i+1)] = model_params["W_"+str(i+1)].dot(model_params["A_"+str(i)]) + model_params["b_"+str(i+1)]
                model_params["A_"+str(i+1)] = self.activation_func(model_params["Z_"+str(i+1)], self.activation_functions[i], derivative=False)

        return model_params

    def backward_prop(self, model_params, train_data, train_labels):
        for i in range(len(self.sizes), 0, -1):
            if i == len(self.sizes):
                if model_params["A_"+str(i)].shape == self.one_hot_encoder(train_labels).shape:
                    model_params["delta_"+str(i)] = model_params["A_"+str(i)] - self.one_hot_encoder(train_labels)
                else:
                    raise Exception("Dimension of the last layer must be the same as the training labels")
            else:
                model_params["delta_"+str(i)] = \
                    model_params["W_"+str(i+1)].T.dot(model_params["delta_"+str(i+1)]) * \
                        self.activation_func(model_params["Z_"+str(i)], self.activation_functions[i-1], derivative=True)
            if i == 1:
                model_params["dW_"+str(i)] = model_params["delta_"+str(i)].dot(train_data.T)
            else:
                model_params["dW_"+str(i)] = model_params["delta_"+str(i)].dot(model_params["A_"+str(i-1)].T)
            
            model_params["db_"+str(i)] = model_params["delta_"+str(i)]
        return model_params

    def one_hot_encoder(self, train_labels):        #tensorized function for performance: to see what is happening see test.ipynb
        Y = np.zeros((int(np.max(train_labels)+1), int(train_labels.shape[0])))
        Y[train_labels, np.arange(train_labels.shape[0])] = 1
        return Y
    


