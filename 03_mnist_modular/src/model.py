import numpy as np
import pandas as pd

class NeuralNetwork():

    activation_functions = []       #Vector with the activation function desired for each layer in sizes: activation_function = ['relu', 'relu', 'sigmoid', ...]

    optimizer: str

    adam_beta1: float
    adam_beta2: float
    adam_eps: float

    train_DATA_PATH: str
    test_DATA_PATH: str                    

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

    def update_params(self, model_params, adam_params, lr, optimizer, iter):
        if optimizer == "gd": #gradient descent
            for i in range(len(self.sizes)):
                model_params["W_"+str(i+1)] = model_params["W_"+str(i+1)] - lr*model_params["dW_"+str(i+1)]
                model_params["b_"+str(i+1)] = model_params["b_"+str(i+1)] - lr*model_params["db_"+str(i+1)]
                return model_params
        if optimizer == "adam": #adam optimizer
            model_params, adam_params = self.adam(model_params, adam_params, lr, iter)
            return model_params, adam_params
        else:
            raise Exception("Optimizer not recognized")

    def adam(self, model_params, adam_params, lr, iter):
        if iter == 1:
            adam_params = self.adam_init(model_params)
        for i in range(len(self.sizes)):
            adam_params["m_W_"+str(i+1)] = self.adam_beta1*adam_params["m_W_"+str(i+1)] + (1-self.adam_beta1)*model_params["dW_"+str(i+1)]
            adam_params["norm_m_W_"+str(i+1)] = adam_params["m_W_"+str(i+1)]/(1-self.adam_beta1)**iter

            adam_params["m_b_"+str(i+1)] = self.adam_beta1*adam_params["m_b_"+str(i+1)] + (1-self.adam_beta1)*model_params["db_"+str(i+1)]
            adam_params["norm_m_b_"+str(i+1)] = adam_params["m_b_"+str(i+1)]/(1-self.adam_beta1)**iter

            adam_params["v_W_"+str(i+1)] = self.adam_beta2*adam_params["v_W_"+str(i+1)] + (1-self.adam_beta2)*(model_params["dW_"+str(i+1)])**2
            adam_params["norm_v_W_"+str(i+1)] = adam_params["v_W_"+str(i+1)]/(1-self.adam_beta2)**iter

            adam_params["v_b_"+str(i+1)] = self.adam_beta2*adam_params["v_b_"+str(i+1)] + (1-self.adam_beta2)*(model_params["db_"+str(i+1)])**2
            adam_params["norm_v_b_"+str(i+1)] = adam_params["v_b_"+str(i+1)]/(1-self.adam_beta2)**iter

            model_params["W_"+str(i+1)] = model_params["W_"+str(i+1)] - \
                lr*adam_params["norm_m_W_"+str(i+1)]/(np.sqrt(adam_params["norm_v_W_"+str(i+1)])+self.adam_eps)
            model_params["b_"+str(i+1)] = model_params["b_"+str(i+1)] - \
                lr*adam_params["norm_m_b_"+str(i+1)]/(np.sqrt(adam_params["norm_v_b_"+str(i+1)])+self.adam_eps)
        return model_params, adam_params

    def adam_init(self,model_params):
        adam_params = {}
        for i in range(len(self.sizes)):
            adam_params["m_W_"+str(i+1)] = np.zeros_like(model_params["W_"+str(i+1)])
            adam_params["v_W_"+str(i+1)] = np.zeros_like(model_params["W_"+str(i+1)])

            adam_params["m_b_"+str(i+1)] = np.zeros_like(model_params["b_"+str(i+1)])
            adam_params["v_b_"+str(i+1)] = np.zeros_like(model_params["b_"+str(i+1)])
        return adam_params


    def train(self, epochs, lr, batch_size, train_type):

        train_data, train_labels, test_data, test_labels = self.data(self.train_DATA_PATH, self.test_DATA_PATH)

        if train_type == "complete":            #model init
            model_params = self.init_model(train_data)
        if train_type == "batch":
            batch_dim = train_data.shape[1] // batch_size
            train_data_batch = train_data[:,int(1*batch_size):int((2)*batch_size)]
            model_params = self.init_model(train_data_batch)

        iter = 1
        for i in range(epochs):

            if train_data == "complete":
                model_params = self.forward_prop(model_params, train_data)
                model_params = self.backward_prop(model_params, train_data, train_labels)
                if self.optimizer == "adam":
                    adam_params = self.adam_init(model_params)
                    model_params = self.update_params(model_params, adam_params, lr, self.optimizer, iter)
                    iter += 1
                else:
                    model_params = self.update_params(model_params, adam_params is None, lr, self.optimizer, iter)

            if train_data == "batch":
                for j in range(batch_dim):
                    train_data_batch = train_data[:,int(j*batch_size):int((j+1)*batch_size)]
                    train_labels_batch = train_labels[int(j*batch_size):int((j+1)*batch_size)]
                    model_params = self.forward_prop(model_params, train_data_batch)
                    model_params = self.backward_prop(model_params, train_data_batch, train_labels_batch)
                    if self.optimizer == "adam":
                        adam_params = self.adam_init(model_params)
                        model_params = self.update_params(model_params, adam_params, lr, self.optimizer, iter)
                        iter =+ 1
                    else:
                        model_params = self.update_params(model_params, adam_params is None, lr, self.optimizer, iter)

        return model_params
                    
            


