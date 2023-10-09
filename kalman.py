'''
    This is a simple kalman filter used to estimate the S&P500's price
    By: Mohammed Sharieff
'''

# Import necessary libraries
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

# Initialize a figure and a plot
fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(121)
ay = fig.add_subplot(122)

# Read and store the dataset
data = pd.read_csv('spdata.csv')

# Calculates Covariace Matrix
def Covar(x):
    x = np.array(x)
    m, n = x.shape
    mu = (1/m)*np.ones(m).dot(x)
    cv = (1/(m-1))*(x - mu).T.dot(x - mu)
    return cv

# Sets a cap on the data length
cap = 150

# Set storage cap (15 days)
lookback = 10

# Extracts the adjusted close
close = data['adjClose'].values.tolist()[:cap]

# Reverses the close prices to oldest to newest and wraps it in a numpy array
close = np.array(close[::-1])

# Compute the velocity and acceleration of the S&P500
velocity = close[1:]/close[:-1] - 1
acceleration = velocity[1:] - velocity[:-1]

# Adjust data lists to be equal
close = close[2:]
velocity = velocity[1:]

# Set a training level and define inputs
T = len(acceleration)

Fk = np.array([[1, 1], [0, 1]])
Bk = np.array([[0.5],[1]])
xk = np.array([[close[0]], [velocity[0]]])
zk = np.array([[0], [0]])
Pk = np.array([[rd.random(), rd.random()], [rd.random(), rd.random()]])
Qk = np.array([[0, 0],[0, 0]])
Hk = np.array([[1, 0],[0, 1]])
Rk = np.array([[0, 0],[0, 0]])
x1k = np.array([[0], [0]])
p1k = np.array([[0, 0],[0, 0]])

store_q, store_z = [], []
plotx, plot_predicted, plot_actual = [], [], []

error = 0
error_graph = []

# Loop till time ends
for t in range(1, T):

    # Prediction Price
    xk = Fk @ xk + Bk*acceleration[t - 1]
    Pk = Fk @ Pk @ Fk.T + Qk

    # Actual Price
    zk = np.array([[close[t]],[velocity[t]]])

    plot_x = range(t)
    plot_predicted.append(xk[0][0])
    plot_actual.append(close[t])

    ax.cla()
    ax.plot(plot_x, plot_predicted, color='limegreen', label='Predicted')
    ax.plot(plot_x, plot_actual, color='red', label='Actual')
    ax.legend()

    error = abs(xk[0][0] - close[t])
    error_graph.append(error)
    title = f'Kalman Filter'
    ax.set_title(title)
    ay.hist(error_graph, bins=10, color='blue')
    ay.set_title("Price Error Distribution")
    ay.set_xlabel('Price Distribution')
    ay.set_ylabel('Frequency')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    
    # Compute and store Covariances
    store_q.append([zk[0][0] - xk[0][0], zk[1][0] - xk[1][0]])
    store_z.append([zk[0][0], zk[1][0]])

    # Handle overflow
    if len(store_q) > lookback:
        del store_q[0]
        del store_z[0]

    if len(store_q) >= 2:
        Qk = Covar(store_q)
        Rk = Covar(store_z)

    # Calculate Kalman Gain
    Kg = Pk @ Hk.T @ np.linalg.inv(Hk @ Pk @ Hk.T + Rk)

    # Update values
    x1k = xk + Kg @ (zk - Hk @ xk)
    p1k = Pk - Kg @ Hk @ Pk

    xk = x1k
    Pk = p1k

    plt.pause(0.0001)


plt.show()
