import numpy as np
import pickle

class neuralnetwork:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, learning_rate=0.12):
        self.params = {}
        np.random.seed(2)
        self.params['W1'] = weight_init_std * np.random.randn(hidden_size, input_size)
        self.params['b1'] = np.zeros((hidden_size, 1))
        self.params['W2'] = weight_init_std * np.random.randn(output_size, hidden_size)
        self.params['b2'] = np.zeros((output_size, 1))

        self.cost = []
        self.m = 0
        self.learning_rate = learning_rate

        self.cache = {}
        self.trained = False

        self.input_size = input_size
        self.hidden_size = hidden_size
        self. output_size = output_size

    def set_learning_rate(learning_rate = 0.12):
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return np.multiply(self.sigmoid(x) , (1 - self.sigmoid(x)))

    def tanh(self, x):
        ### Your code here (1-2 lines) ###
        return None

    def tanh_prime(self, x):
        ### Your code here (1-2 lines) ###
        return None


    def forward(self, X): 
        """
        computes forward propagation
        """
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2'] 

        # Forward propagation 
        ### Your code here (4 lines) ###
        Z1 = np.dot(W1, X) + b1 
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2) 

        # save Z1, A1, Z2 and A2 in cache for later use
        self.cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    def cross_entropy_loss(self, Y_hat):
        """
        computes cross entropy loss function
        input: Y_hat, vector of results predicted by our neural network
        output: J, cost value of Y_hat
        """
        A2 = self.cache["A2"]
        assert self.m != 0

        ### Your code here (~1-2 lines) ###
        
        # repack J into a scalar, e.g. not [[19.0]]
        J = np.squeeze(J)
        # save J in a list for plotting loss function graph
        self.cost.append(J)
        return J
        

    def backward(self, X, Y):
        """
        computes back propagation
        """
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2'] 
        Z1 = self.cache['Z1']
        A1 = self.cache["A1"]
        A2 = self.cache["A2"]

        assert self.m != 0

        # Backward propagation
        ### Your code here (6 lines) ###
        dZ2 =
        dW2 =
        db2 = 
        dZ1 = 
        dW1 = 
        db1 = 

        return {"dW1":dW1, "db1":db1, "dW2":dW2, "db2":db2}

    def update(self, gradients):
        """
        update the parameters of W1, b1, W2 and b2 using the gradients 
        computed from back propagation and the learning rate alpha
        input: gradients dictionary from back propagation
        """
        # load parameters
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2'] 
        # load gradients from back propagation
        dW1, db1, dW2, db2 = gradients['dW1'], gradients['db1'], gradients['dW2'], gradients['db2'],

        # Update Parameters
        ### Your code here (4 lines) ###
        W1 = 
        b1 = 
        W2 = 
        b2 =        
        
        self.params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def train(self, X, Y, m, num_iterations = 20000, debug = False):
        self.cost = []
        self.m = m
        assert self.m != 0

        print("Training start. Using learing rate %f"%(self.learning_rate))
        print("="*50)

        for i in range(0, num_iterations):
            self.forward(X)
            J = self.cross_entropy_loss(Y)
            gradients = self.backward(X, Y)
            self.update(gradients)

            self.cost.append(J)
            if debug and i%500==0:
                print("cost after iteration %i: %2.8f"%(i, J))

        print("Training is completed!")
        self.trained = True

    def predict(self, X):
        if self.trained:
            self.forward(X)
            prediction = (self.cache["A2"] > 0.5)
            print("Output is generated.")
            return prediction
        else:
            print("Please train the network first!")
            return None

    def output(self):
        if self.trained:
            out = open('nn_weights.dat', 'wb')
            pickle.dump(self.params, out)
        else:
            print("Please train the network with data first!") 

    def input(self, weight_file):
        self.params = pickle.load(weight_file)
        self.trained = True

    def getsize(self):
        return self.input_size, self.hidden_size, self.output_size

    def getcostlist(self):
        return self.cost
