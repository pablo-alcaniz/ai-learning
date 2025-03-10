import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

class NeuralNetwork():

    activation_functions = []       #Vector with the activation function desired for each layer in sizes: activation_function = ['relu', 'relu', 'sigmoid', ...]

    optimizer: str

    adam_beta1: float
    adam_beta2: float
    adam_eps: float

    train_DATA_PATH: str
    test_DATA_PATH = None

    batch_prints = 4 #Default batch prints

    SAVE_DIR: str
    SAVE_NAME: str

    LOAD_DIR: str
    LOAD_NAME: str                    

    def __init__(self, sizes):
        self.sizes = sizes      #sizes of the layers including outputs but excluding inputs               

    def data(self):

        train_data_raw = pd.read_csv(self.train_DATA_PATH)               #read csv file with pandas
        train_data_raw = train_data_raw.fillna(0)                   #prevent NaN
        train_data_raw = np.array(train_data_raw)                   #convert pandas type to a numpy one
        np.random.shuffle(train_data_raw)                           #shuffle data to prevent overfitting
        train_data = train_data_raw[:,1:]                           #split the train data from the labels
        train_data = np.transpose(train_data/train_data.max())      
        train_labels = train_data_raw[:,0]                          #take the labels for the training

        if self.test_DATA_PATH is None:
            test_data = None
            test_labels = None
            print("INFO: Data loaded correctly")
            print("WARNING: Only train data loaded")
            return train_data, train_labels, test_data, test_labels                   #if user doesn't specifies a test set return the train data 
        else:
            test_data_raw = pd.read_csv(self.test_DATA_PATH)
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
            model_params["b_"+str(i+1)] = np.random.rand(self.sizes[i],1) - 0.5                     #bias init
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
        if func == "softmax": #stable softmax
            if derivative is False:
                Z_stable = Z - np.max(Z, axis=0, keepdims=True)
                exp_Z = np.exp(Z_stable)
                softmax_output = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
                return softmax_output
            if derivative is True:
                raise Exception("Softmax derivative its not implemented yet. Please use another.")
        else:
            raise Exception("Activation function not recognized")

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
                model_params["dW_"+str(i)] = model_params["delta_"+str(i)].dot(train_data.T)/train_data.shape[1] #VERY IMPORTANT TO NORMALIZE dW WITH THE NUMBER OF PARAMETERS
            else:
                model_params["dW_"+str(i)] = model_params["delta_"+str(i)].dot(model_params["A_"+str(i-1)].T)/train_data.shape[1]
            
            model_params["db_"+str(i)] = np.mean(model_params["delta_"+str(i)], axis=1, keepdims=True) #we need to do this to acomodate the dimensions

        return model_params
    
    def one_hot_encoder(self, train_labels):        #tensorized function for performance: to see what is happening see test.ipynb
        train_labels = train_labels.astype(int)
        if int(np.max(train_labels)+1) < 10:        #WARNING: when batch size is short enough there a are a probabiliy that there is no 9, so the dimension result is <10, so it will raise an exception in backprop function
            Y = np.zeros((int(10), int(train_labels.shape[0])))
        else:
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
        for i in range(len(self.sizes)):
            adam_params["m_W_"+str(i+1)] = self.adam_beta1*adam_params["m_W_"+str(i+1)] + (1-self.adam_beta1)*model_params["dW_"+str(i+1)]
            adam_params["norm_m_W_"+str(i+1)] = adam_params["m_W_"+str(i+1)]/(1-self.adam_beta1**iter)

            adam_params["m_b_"+str(i+1)] = self.adam_beta1*adam_params["m_b_"+str(i+1)] + (1-self.adam_beta1)*model_params["db_"+str(i+1)]
            adam_params["norm_m_b_"+str(i+1)] = adam_params["m_b_"+str(i+1)]/(1-self.adam_beta1**iter)

            adam_params["v_W_"+str(i+1)] = self.adam_beta2*adam_params["v_W_"+str(i+1)] + (1-self.adam_beta2)*(model_params["dW_"+str(i+1)])**2
            adam_params["norm_v_W_"+str(i+1)] = adam_params["v_W_"+str(i+1)]/(1-self.adam_beta2**iter)

            adam_params["v_b_"+str(i+1)] = self.adam_beta2*adam_params["v_b_"+str(i+1)] + (1-self.adam_beta2)*(model_params["db_"+str(i+1)])**2
            adam_params["norm_v_b_"+str(i+1)] = adam_params["v_b_"+str(i+1)]/(1-self.adam_beta2**iter)

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
    
    def precision(self, model_params, true_data):
        model_prediction = np.argmax(model_params["A_"+str(len(self.sizes))], axis=0)
        return np.mean(model_prediction == true_data)
        
        
    def train(self, epochs, lr, batch_size, train_type, log):
        
        tic_data = time.perf_counter()
        train_data, train_labels, test_data, test_labels = self.data()
        toc_data = time.perf_counter()
        print("Data load time: ", toc_data-tic_data,"s")

        tic = time.perf_counter()

        if train_type == "complete":            #model init
            model_params = self.init_model(train_data)
        if train_type == "batch":
            batch_dim = train_data.shape[1] // batch_size
            train_data_batch = train_data[:,int(0*batch_size):int(batch_size)]
            model_params = self.init_model(train_data_batch)

        iter = 1
        for i in range(epochs):

            if train_type == "complete":
                model_params = self.forward_prop(model_params, train_data)
                model_params = self.backward_prop(model_params, train_data, train_labels)
                if self.optimizer == "adam":
                    adam_params = self.adam_init(model_params)
                    model_params, adam_params = self.update_params(model_params, adam_params, lr, self.optimizer, iter)
                    iter += 1
                else:
                    adam_params = None
                    model_params = self.update_params(model_params, adam_params, lr, self.optimizer, iter)
                if log == True:
                    if i % 20 == 0:
                        print("Epoch: ", i)
                        print("Estimated precision: ", self.precision(model_params, train_labels))
            if train_type == "batch":
                for j in range(batch_dim):
                    train_data_batch = train_data[:,int(j*batch_size):int((j+1)*batch_size)]
                    train_labels_batch = train_labels[int(j*batch_size):int((j+1)*batch_size)]
                    model_params = self.forward_prop(model_params, train_data_batch)
                    model_params = self.backward_prop(model_params, train_data_batch, train_labels_batch)
                    if self.optimizer == "adam":
                        adam_params = self.adam_init(model_params)
                        model_params, adam_params = self.update_params(model_params, adam_params, lr, self.optimizer, iter)
                        iter += 1
                    else:
                        adam_params = None
                        model_params = self.update_params(model_params, adam_params, lr, self.optimizer, iter)

                    if self.batch_prints == 0:
                        log_batch = False
                    else:
                        log_batch = True
                        printable = max(1, batch_dim // self.batch_prints)
                    if log_batch == True and log == True:
                        if j % printable == 0:
                            print("Epoch: ", i)
                            print("Batch: ", j,"/",batch_dim)
                            print("Estimated precision: ", self.precision(model_params, train_labels_batch))

        toc = time.perf_counter()
        if log == True:
            print("---Epochs: ", epochs)
            print("---Batch size: ", batch_size)
            print("---Learning rate: ", lr)
            print("---Optimizer: ", self.optimizer)
            if self.test_DATA_PATH is not None:
                print("---Model precision: ", self.test_full_model(model_params,test_data,test_labels))
            print("---Time of training: ", toc-tic,"s")
        return model_params
    
    def save_model(self, model_params):
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        MODEL_SAVE_DIR = os.path.join(self.SAVE_DIR, self.SAVE_NAME)
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

        for i in range(len(self.sizes)):
            np.save(os.path.join(MODEL_SAVE_DIR,"W_"+str(i+1)+".npy"), model_params["W_"+str(i+1)])
            np.save(os.path.join(MODEL_SAVE_DIR,"b_"+str(i+1)+".npy"), model_params["b_"+str(i+1)])
            np.save(os.path.join(MODEL_SAVE_DIR,"activation_funcs.npy"), self.activation_functions)

    def load_model(self):
        MODEL_LOAD_DIR = os.path.join(self.LOAD_DIR, self.LOAD_NAME)
        model_params = {}

        for i in range(len(self.sizes)):
            model_params["W_"+str(i+1)] = np.load(os.path.join(MODEL_LOAD_DIR,"W_"+str(i+1)+".npy"))
            model_params["b_"+str(i+1)] = np.load(os.path.join(MODEL_LOAD_DIR,"b_"+str(i+1)+".npy"))
            self.activation_functions = np.load(os.path.join(MODEL_LOAD_DIR,"activation_funcs.npy"))
        return model_params
    
    def test_full_model(self, model_params, test_data, test_labels):
        model_params = self.forward_prop(model_params, test_data)
        return self.precision(model_params,test_labels)
    
    def plot_image(self, case, data, data_labels):
        image = np.array(data[:,case]).reshape(int(np.sqrt(data.shape[0])),int(np.sqrt(data.shape[0])))
        plt.imshow(image, cmap='gray')
        plt.title(f'Value: {data_labels[case]}')
        plt.show()

    def test_prediction(self, case, data, data_label, model_params):

        if "A_"+str(len(self.sizes)) in model_params:
            model_params.pop("A_"+str(len(self.sizes)))
        
        data_vector = np.zeros((len(data[:,case]),1))
        for i in range(len(data[:,case])):
            data_vector[i] = data[i,case]

        model_params = self.forward_prop(model_params, data_vector) #it has to be pass as a vector because potato (some shit of broadcasting or idk waht)
        model_prediction = np.argmax(model_params["A_"+str(len(self.sizes))], axis=0)
        real_data = data_label[case]

        self.plot_image(case, data, data_label)

        print("Model Prediction: ", model_prediction)
        print("Real value: ", real_data)
        
