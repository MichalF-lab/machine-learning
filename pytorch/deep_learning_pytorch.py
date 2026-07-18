# %load mnist_loader.py
"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e




from RidgeRegr import RidgeRegr
import torch.nn as nn

class Network(nn.Module):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]


    def SGD(self, training_data, epochs, mini_batch_size, alpha, test_data=None):
        training_data = list(training_data)
        n = len(training_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
                ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, alpha):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # o ile ma sie zmienic wagi i biasy
            delta_gradient_b, delta_gradient_w = self.backprop(x, y)
            gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
            gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]

        # zmiana wag i biasow
        self.biases = [b-(alpha/len(mini_batch))*nb for b, nb in zip(self.biases, gradient_b)]
        self.weights = [w-(alpha/len(mini_batch))*nw for w, nw in zip(self.weights, gradient_w)]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a


    def backprop(self, x, y):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        neuron = x
        neural_network = [x]
        z_tab = []

        # Forward
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, neuron)+b
            z_tab.append(z)
            neuron = self.sigmoid(z)
            neural_network.append(neuron)

        # Backward
        gradient_b[-1] = self.cost_derivative(neural_network[-1], y) * self.sigmoid_prime(z_tab[-1])
        gradient_w[-1] = np.dot(gradient_b[-1], neural_network[-2].transpose())

        #print(gradient_b[-1][0], gradient_w[-1][0],"\n")

        for layer in range(2, self.num_layers):
            z = z_tab[-layer]
            gradient_b[-layer] = np.dot(self.weights[-layer+1].transpose(), gradient_b[-layer+1]) * self.sigmoid_prime(z)
            gradient_w[-layer] = np.dot(gradient_b[-layer], neural_network[-layer-1].transpose())
        return (gradient_b, gradient_w)

    def ridge_fit(self, training_data, alpha=0.2, learning_rate=0.0005):

        train_list = list(training_data)

        # Build feature matrix X (n_samples, hidden_size) and target matrix Y (n_samples, n_outputs)
        X_rows = []
        Y_rows = []
        for x, y in train_list:
            a = x
            # propagate up to last hidden layer (exclude final layer)
            for b, w in zip(self.biases[:-1], self.weights[:-1]):
                a = self.sigmoid(np.dot(w, a) + b)
            X_rows.append(a.ravel())
            Y_rows.append(y.ravel())

        X = np.vstack(X_rows)  # shape (n_samples, hidden_size)
        Y = np.vstack(Y_rows)  # shape (n_samples, n_outputs)

        hidden_size = X.shape[1]
        n_outputs = Y.shape[1]

        # Fit ridge for each output neuron
        new_W = np.zeros((n_outputs, hidden_size))
        new_b = np.zeros((n_outputs, 1))
        for j in range(n_outputs):
            rr = RidgeRegr(alpha=alpha)
            rr.fit(X, Y[:, j], learning_rate=learning_rate)
            theta = rr.theta  # shape (hidden_size + 1,)
            new_b[j, 0] = theta[0]
            new_W[j, :] = theta[1:]

        # Replace final layer weights and biases
        self.weights[-1] = new_W
        self.biases[-1] = new_b

    def evaluate(self, test_data): # procent correct
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))


training_data, validation_data, test_data = load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)

net1 = Network([784, 30, 10])

net1.SGD(training_data, 10, 10, 3.0, test_data=test_data)
accuracy = net1.evaluate(validation_data)
print(f"Validation Accuracy: {accuracy} / {len(validation_data)}")


net2 = Network([784, 30, 10])

net2.ridge_fit(training_data, alpha=0.3, learning_rate=0.001)
accuracy = net2.evaluate(validation_data)
print(f"Validation Accuracy: {accuracy} / {len(validation_data)}")