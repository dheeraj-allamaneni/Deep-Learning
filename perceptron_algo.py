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

    w = np.zeros(len(X[0]))
    eta = 1
    n = 30
    errors = []

    for t in range(n):
        total_error = 0
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                total_error += (np.dot(X[i], w)*Y[i])
                w = w + eta*X[i]*Y[i]
        errors.append(total_error*-1)
        
    # plt.plot(errors)
    # plt.xlabel('Epoch')
    # plt.ylabel('Total Loss')
    
    return w
w=perceptron_algo(X,y)
print(perceptron_algo(X,y))

for d, sample in enumerate(X):
    # Ploting the negative sample data
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Ploting the positive sample data
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)
plt.plot([1.61,1.27],[4.57,1.099])

plt.show()
