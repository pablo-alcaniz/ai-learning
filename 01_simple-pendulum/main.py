import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

## FUNCTIONS ##
def euler(theta_n, omega_n, h, freq):
    theta_n1 = theta_n + h*omega_n
    omega_n1 = (theta_n1-theta_n)/h - h*freq*np.sin(theta_n)
    return theta_n1, omega_n1

## PHYSICS PROBLEM PARAMETERS ##

g = 9.81 #m/s^2
L = float(1.0) #m
freq = np.square(g/L)

t_f = 10 #s
t_0 = 0 #s
n = 100000 #divisions
t = np.linspace(t_0, t_f, n)

h = (t_f-t_0)/n #step

theta_0 = np.pi/4
omega_0 = 0

THETA = np.zeros(t.size)
OMEGA = np.zeros(t.size)

THETA[0] = theta_0
OMEGA[0] = omega_0

## PHYSICS PROBLEM SOLUTION
i = 1
while i < t.size:
   THETA[i], OMEGA[i] = euler(THETA[i-1], OMEGA[i-1],h,freq) 
   i += 1 

plt.plot(t,THETA)
#plt.show()

## AUTOENCODER ## 

X = np.column_stack((THETA, OMEGA))

#data normalization
X_min, X_max = X.min(), X.max()
X = (X - X_min) / (X_max - X_min)

#autoencoder definition
encoding_dim = 1 
input_layer = Input(shape=(2,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(2, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

#autoencoder train
autoencoder.fit(X, X, epochs=200, batch_size=10, verbose=1)

#reconstruction
X_reconstructed = autoencoder.predict(X)
X_reconstructed = X_reconstructed * (X_max - X_min) + X_min  # Desnormalizar

#results

# Graficar resultados
plt.figure(figsize=(10, 4))
plt.plot(t, THETA, label='Theta original', linestyle='dashed')
plt.plot(t, X_reconstructed[:, 0], label='Theta reconstruida', linestyle='solid')
plt.legend()
plt.xlabel("Tiempo (s)")
plt.ylabel("Ángulo (rad)")
plt.title("Reconstrucción del movimiento del péndulo con Autoencoder")
plt.show()