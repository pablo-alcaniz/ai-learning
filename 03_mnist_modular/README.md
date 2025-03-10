# Complete Documentation of the Modular Neural Network Model for MNIST

This documentation explains in detail what to do and how to do it in order to use the neural network model defined in `model.py` and run it via `main.py`. It covers data preparation, model configuration, training, evaluation, saving/loading the model, and result visualization, including a basic usage example.

---

## Requirements

- **Python 3.x**
- **Required Libraries**:
  - NumPy
  - Pandas
  - Matplotlib
  - (Standard libraries: `os`, `time`)

- **Data**:
  - A training CSV file (e.g., `mnist_train.csv`)
  - A test CSV file (e.g., `mnist_test.csv`)

> **Note**: The CSV files should have the following format:  
> The first column contains the label (for example, digits 0 through 9), and the remaining columns contain pixel values of the images.

---

## Project Structure

- **`model.py`**  
  Contains the `NeuralNetwork` class, which includes:
  - Functions to load and preprocess data.
  - Methods to initialize model parameters (weights and biases).
  - Multiple activation functions (ReLU, Sigmoid, Tanh, Softmax).
  - Forward and backward propagation methods (`forward_prop` and `backward_prop`).
  - Parameter updates (GD or Adam).
  - Methods to evaluate accuracy, save, and load the model.
  - Utility functions to visualize images (`plot_image`) and test single predictions (`test_prediction`).

- **`main.py`**  
  A sample script that configures model parameters, loads data, trains the network, and saves the trained model.

- **(Optional) `test.ipynb`**  
  A Jupyter notebook for interactive tests and visualizations.

---

## Data Preparation

1. **CSV Format and Preprocessing**:
   - The CSV file must have the first column as the label and the subsequent columns as pixel values.
   - The code uses `pandas` to read CSV files, filling missing values with 0 and converting the data to a NumPy array.
   - The pixel values are normalized by dividing by the maximum pixel value.
   - The data is then transposed so that each column corresponds to one training example.

2. **Paths**:
   - In `main.py`, configure the file paths for training and test data, for example:
     ```python
     nn.train_DATA_PATH = "src/data/mnist_train.csv"
     nn.test_DATA_PATH  = "src/data/mnist_test.csv"
     ```

---

## Model Configuration

1. **Defining the Architecture**:
   - Set the `sizes` parameter to define the number of neurons for each layer (excluding the input layer).  
     Example:
     ```python
     sizes = [20, 10]
     ```
     This implies one hidden layer with 20 neurons and an output layer with 10 neurons (for 10 classes).

2. **Activation Functions**:
   - In the list `activation_functions`, specify which activation to use for each layer:
     ```python
     nn.activation_functions = ["relu", "sigmoid"]
     ```

3. **Adam Optimizer Parameters**:
   - Configure Adam hyperparameters:
     ```python
     nn.adam_beta1 = 0.9
     nn.adam_beta2 = 0.99
     nn.adam_eps   = 1E-8
     nn.optimizer  = "adam"
     ```

4. **Other Parameters**:
   - `batch_prints` determines how often to print logs during training.
   - `train_type` can be `"complete"` to process all data at once or `"batch"` to train in mini-batches.
   - Example:
     ```python
     nn.batch_prints = 5
     ```

---

## Training the Model

The `train` method of the `NeuralNetwork` class follows these steps:

1. **Load and Preprocess Data**:
   - Calls the `data()` method to read and normalize the CSV files. It produces:
     - `train_data`, `train_labels`
     - `test_data`, `test_labels` (if a test path is provided)

2. **Initialize Parameters**:
   - Uses `init_model()` to randomly initialize weights and biases based on the dimensions of the training data (or the first batch in batch mode).

3. **Forward and Backward Propagation**:
   - **Forward Propagation**: Calculates each layer’s activation using the configured activation functions (`forward_prop`).
   - **Backward Propagation**: Computes gradients using the method `backward_prop()` and one-hot encoding of the labels.

4. **Parameter Updates**:
   - Updates weights and biases via `update_params()`, using either gradient descent or Adam.

5. **Logging and Evaluation**:
   - Logs are printed at intervals during the training, showing the accuracy on the current batch or epoch.
   - At the end, if a test dataset is available, the model’s final accuracy is reported.

6. **Training Call Example**:
   ```python
   params = nn.train(epochs=1, lr=0.01, batch_size=128, train_type="batch", log=True)
