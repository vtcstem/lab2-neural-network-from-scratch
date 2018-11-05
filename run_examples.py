import numpy as np
import matplotlib.pyplot as plt
import pylab
import sklearn
import sklearn.datasets
import sklearn.linear_model
import pickle
import sys
import os.path
from argparse import ArgumentParser
from neuralnetwork import *


def layer_size(X,Y):
    """
        Arguments:
        X -- input dataset of shape (input size, number of examples)
        Y -- labels of shape (output size, number of examples)

        Returns:
        n_x -- the size of the input layer
        n_h -- the size of the hidden layer
        n_y -- the size of the output layer
    """
    n_x=X.shape[0]
    n_h= 6
    n_y=Y.shape[0]
    return (n_x,n_h,n_y)

def load_planar_dataset():
    np.random.seed(1)
    m = 400  
    N = int(m / 2)  
    D = 2 
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T
    plt.scatter(X[0, :], X[1, :],c=Y.reshape(X[0,:].shape),  s=40, cmap=plt.cm.Spectral);

    return X, Y

def load_moon_dataset():
    np.random.seed(0)
    X, Y = sklearn.datasets.make_moons(200, noise=0.20)
    plt.scatter(X[:,0], X[:,1], s=40, c=Y, cmap=plt.cm.Spectral)
    return X.T, np.reshape(Y, (1, Y.shape[0]))

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.reshape(X[0,:].shape), cmap=plt.cm.Spectral)

parser = ArgumentParser()
parser.add_argument("--output", help="output trained weights", action="store_true")
parser.add_argument("-d", "--dataset", help="choose dataset: moon / planar", dest="dataset", default="moon")
parser.add_argument("-lr", "--learningrate", help="define the learning rate", dest="lr", type=float, default=0.12)
parser.add_argument("-i", "--iteration", help="define the learning rate", dest="iter", type=int, default=20000)
args = parser.parse_args()

if str(args.dataset) == "moon":
    X, Y = load_moon_dataset()
    #print(args.dataset)
elif str(args.dataset) == "planar":
    X, Y = load_planar_dataset()
else:
    print("Unrecognized dataset parameter! Exit now.")
    sys.exit(1)


plt.show()

shape_X=X.shape
shape_Y=Y.shape
m=X.shape[1]
print("the shape of X is:"+str(shape_X))
print("the shape of Y is:"+str(shape_Y))
print("the training examples:" +str(m))

n_x, n_h, n_y = layer_size(X,Y)

nn = neuralnetwork(n_x, n_h, n_y, learning_rate=args.lr)

if os.path.isfile('nn_weights.dat'):
    print("Found existing weights. Loading...")
    weights = open('nn_weights.dat', 'rb')
    nn.input(weights)
else:
    print("Train new weights using data...")
    nn.train(X, Y, m, num_iterations = args.iter, debug = True)

plot_decision_boundary(lambda x: nn.predict(x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(nn.getsize()[1]))
plt.show()


print ('Plotting cost function over iteration')
cost = nn.getcostlist()
plt.plot(cost)
plt.title("Cost function over iteration")
plt.ylabel('cost')
plt.xlabel('iteration')
plt.show()


predictions=nn.predict(X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

if args.output:
    print ('Saving weights...')
    nn.output()



