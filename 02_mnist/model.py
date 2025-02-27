import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def import_data():
    train_data_raw = pd.read_csv('data-set/mnist_train.csv')
    train_data_raw = np.array(train_data_raw)
    train_data = train_data_raw[:,1:]
    train_data = np.transpose(train_data)
    train_labels = train_data_raw[:,0]

    test_data_raw = pd.read_csv('data-set/mnist_test.csv')
    test_data_raw = np.array(test_data_raw) 
    test_data = test_data_raw[:,1:]
    test_data = np.transpose(test_data)
    test_labels = test_data_raw[:,0]

    return train_data, train_labels, test_data, test_labels

def init_model(n1, n2, m):
    W1 = np.random.rand(n1,784) - 0.5
    b1 = np.random.rand(n1,m) - 0.5

    W2 = np.random.rand(n2,n1) - 0.5
    b2 = np.random.rand(n2,m) - 0.5
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
    db2 = 1/m * np.sum(dZ2,1)
    dZ1 = np.transpose(W2).dot(dZ2) * ReLU_diff(Z1)
    dW1 = 1/m * dZ1.dot(np.transpose(X))
    db1 = 1/m * np.sum(dZ1,1)
    return dW1, db1, dW2, db2

def update(W1, dW1, b1, db1, W2, dW2, b2, db2, alpha):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    return W1, b1, W2, b2

def train(X, train_labels, alpha, iterations, n1, n2, m):
    W1, b1, W2, b2 = init_model(n1, n2, m)
    i = 0
    while i < iterations:
        i += 1
        Z1, A1, Z2, A2 = front_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = back_propagation(X, Z1, A1, A2, W2, m, train_labels)
        W1, b1, W2, b2 = update(W1, dW1, b1, db1, W2, dW2, b2, db2, alpha)
        if i % 20 == 0:
            print("Iteration: ", i)
            print("Precision: ", precision(prediction(A2),train_labels))

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

def ideal_output(train_labels):
    Y = np.zeros(train_labels.size, train_labels.max()+1)
    Y[np.arange(train_labels.size), train_labels] = 1
    return np.transpose(Y)

def plot_image_data_set(label, data, labels):
    image = np.array(data[label]).reshape(28,28)
    plt.imshow(image, cmap='gray')
    plt.title(f'Etiqueta: {labels[label]}')
    plt.show()

