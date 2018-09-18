import numpy as np
import matplotlib.pyplot as plt


X = np.array([
	[1, 2, -1],
	[1,4,-1],
	[2,2,-1],
	[4,2,-1],
	[3,4,-1],
	[2,3,-1]
	])
y = np.array([1,1,1,-1,-1,-1])

def perceptron_algo(X, Y):

    w = np.zeros(len(X[0]))# initiating w to zeros as per the given problem *remember this weight vector also includes bias term so the initialization is correct.
    eta = 1
    n = 30
    errors = []

    for t in range(n):
        tot_err = 0
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                tot_err += (np.dot(X[i], w)*Y[i])
                w = w + eta*X[i]*Y[i]
        errors.append(tot_err*-1)
        
    # plt.plot(error's)
    # plt.xlabel('Epoch')
    # plt.ylabel('Total los')
    #This weight vector also includes the bias term
    return w # this will return -13 1 -17
w=perceptron_algo(X,y)
print(perceptron_algo(X,y))

for d, sample in enumerate(X):
    # Ploting the -ve sample data
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Ploting the +ve sample data
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)
plt.plot([1.61,1.27],[4.57,1.099])

plt.show()
