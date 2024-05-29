# copy of the code from http://neuralnetworksanddeeplearning.com/chap1.html
# Added code to dump the initialisation weights and biases, and the mini
# batches into JSON files, for nn_1.pl to pick up and use.  The reason for
# this is to check that all of the calculations in feedforward and backprop
# in Perl match the Python output.
import pickle
import json
import codecs
import numpy as np
import mnist_loader
import random

class Network(object):

    def __init__(self, sizes):
       self.num_layers = len(sizes)
       self.sizes = sizes
       self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
       self.weights = [np.random.randn(y, x) 
                       for x, y in zip(sizes[:-1], sizes[1:])]
       b = []
       for i in self.biases:
          b.append(i.tolist())
       file_path = "bias.json" ## your path variable
       json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), 
           separators=(',', ':'), 
           sort_keys=True, 
           indent=4) ### this saves the array in .json format
       w = []
       for i in self.weights:
          w.append(i.tolist())
       file_path = "weights.json" ## your path variable
       json.dump(w, codecs.open(file_path, 'w', encoding='utf-8'), 
           separators=(',', ':'), 
           sort_keys=True, 
           indent=4) ### this saves the array in .json format

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta, batchno):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        ctr = 0
        for b in self.biases:
           print("self.biases",ctr,"at end of batch", batchno)
           ctr = ctr + 1
           print( b )
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        print("self.weights[-1] at end of batch",batchno)
        print(self.weights[-1])

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            z = 1
            for mini_batch in mini_batches:
                file_path= "mini_batch_e" + str(j).zfill(4) + "_b" + str(z).zfill(5)
                mb = []
                for i in mini_batch:
                   mb.append((i[0].tolist(),i[1].tolist()))
                json.dump(mb, codecs.open(file_path, 'w', encoding='utf-8'),
                   separators=(',', ':'),
                   sort_keys=True,
                   indent=4) ### this saves the array in .json format
                self.update_mini_batch(mini_batch, eta,z)
                z = z + 1
                if z == 100:
                   print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
                   raise SystemExit(0)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))
            raise SystemExit(0)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives partial C_x 
        partial a for the output activations."""
        return (output_activations-y)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784,30,10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
