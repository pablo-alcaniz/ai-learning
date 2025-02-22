import numpy as np
import matplotlib.pyplot as plt

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
plt.show()

    

