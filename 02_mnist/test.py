import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import model as md
import os
import re

raw_data = pd.read_csv('data-set/mnist_test.csv')

raw_data = np.array(raw_data)

labels = raw_data[:,0]
data = raw_data[:,1:]
label=5

a = np.arange(10)
print(a)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = TOKEN_PATH = os.path.join(BASE_DIR, 'data-save')
os.makedirs(SAVE_DIR, exist_ok=True)

np.save(os.path.join(SAVE_DIR, "data.npy"), a)
a = 0
print(a)

a = np.load(os.path.join(SAVE_DIR, "data.npy"))
print(a)