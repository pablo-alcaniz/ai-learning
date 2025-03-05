import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



def import_data():
    train_data_raw = pd.read_csv('data-set/mnist_train.csv')
    train_data_raw = train_data_raw.fillna(0)
    train_data_raw = np.array(train_data_raw)
    np.random.shuffle(train_data_raw)
    train_data = train_data_raw[:,1:]
    train_data = np.transpose(train_data/255)
    train_labels = train_data_raw[:,0]
    test_data_raw = pd.read_csv('data-set/mnist_test.csv')
    test_data_raw = np.array(test_data_raw) 
    test_data = test_data_raw[:,1:]
    test_data = np.transpose(test_data)
    test_labels = test_data_raw[:,0]
    return train_data, train_labels, test_data, test_labels

def init_model(n1, n2,m):
    W1 = np.random.rand(n1,784) - 0.5
    b1 = np.random.rand(n1,1) - 0.5

    W2 = np.random.rand(n2,n1) - 0.5
    b2 = np.random.rand(n2,1) - 0.5
    print('Model init correct.')
    return W1, b1, W2, b2

def front_propagation(X, W1, b1, W2, b2):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def back_propagation(X, Z1, A1, A2, W2, m, train_labels):
    dZ2 = A2 - ideal_output(train_labels)
    dW2 = 1/m * dZ2.dot(np.transpose(A1))
    db2 = 1/m * np.sum(dZ2,1,keepdims=True)
    dZ1 = np.transpose(W2).dot(dZ2) * ReLU_diff(Z1)
    dW1 = 1/m * dZ1.dot(np.transpose(X))
    db1 = 1/m * np.sum(dZ1,1,keepdims=True)
    return dW1, db1, dW2, db2

def update(W1, dW1, b1, db1, W2, dW2, b2, db2, alpha, n1, n2):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    return W1, b1, W2, b2

def train(alpha, iterations, n1, n2, save_name):
    X, train_labels, test_data, test_labels = import_data()
    m = X.shape[1]
    W1, b1, W2, b2 = init_model(n1, n2, m)
    i = 0
    while i < iterations:
        i += 1
        Z1, A1, Z2, A2 = front_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = back_propagation(X, Z1, A1, A2, W2, m, train_labels)
        W1, b1, W2, b2 = update(W1, dW1, b1, db1, W2, dW2, b2, db2, alpha, n1, n2)
        if i % 1 == 0:
            print("Iteration: ", i)
            print("Precision: ", precision(prediction(A2),train_labels))
    save_model_train(save_name, W1, b1, W2, b2)
    print("Final estimated precision:", precision(prediction(A2),train_labels))

def ReLU(Z):
    return np.maximum(Z,0)

def ReLU_diff(Z):
    return Z > 0

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A 

def prediction(A2):
    return np.argmax(A2,0)

def precision(prediction, train_labels):
    return np.sum(prediction == train_labels)/train_labels.size

def save_model_train(save_name, W1, b1, W2, b2):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(BASE_DIR, 'data-save')
    os.makedirs(SAVE_DIR, exist_ok=True)
    MODEL_SAVE_DIR = os.path.join(SAVE_DIR, save_name)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    np.save(os.path.join(MODEL_SAVE_DIR,"W1.npy"), W1)
    np.save(os.path.join(MODEL_SAVE_DIR,"b1.npy"), b1)
    np.save(os.path.join(MODEL_SAVE_DIR,"W2.npy"), W2)
    np.save(os.path.join(MODEL_SAVE_DIR,"b2.npy"), b2)
    print("Parameters of the model saved.")

def load_model(RELATIVE_VAR_PATH):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(BASE_DIR, 'data-save')
    os.makedirs(SAVE_DIR, exist_ok=True)
    MODEL_SAVE_DIR = os.path.join(SAVE_DIR, RELATIVE_VAR_PATH)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    W1 = np.load(os.path.join(MODEL_SAVE_DIR,"W1.npy"), W1)
    b1 = np.load(os.path.join(MODEL_SAVE_DIR,"b1.npy"), b1)
    W2 = np.load(os.path.join(MODEL_SAVE_DIR,"W2.npy"), W2)
    b2 = np.load(os.path.join(MODEL_SAVE_DIR,"b2.npy"), b2)
    print("Parameters of the models loaded.")
    return W1, b1, W2, b2

def ideal_output(train_labels):
    labels = train_labels.astype(int)
    num_classes = 10
    num_examples = labels.size

    Y = np.zeros((num_classes, num_examples))
    for i in range(num_examples):
        if 0 <= labels[i] < num_classes:  
            Y[labels[i], i] = 1
    return Y

def plot_image_data_set(label, data, labels):
    image = np.array(data[label]).reshape(28,28)
    plt.imshow(image, cmap='gray')
    plt.title(f'Etiqueta: {labels[label]}')
    plt.show()