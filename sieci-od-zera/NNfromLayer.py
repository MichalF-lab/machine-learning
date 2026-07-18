import pickle
import gzip

# Third-party libraries
import numpy as np
import cupy as cp


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

def load_data_wrapper_cupy():
    training_data, validation_data, test_data = load_data_wrapper()
    
    # Convert to CuPy arrays
    training_data = [(cp.asarray(x), cp.asarray(y)) for x, y in training_data]
    validation_data = [(cp.asarray(x), y) for x, y in validation_data]
    test_data = [(cp.asarray(x), y) for x, y in test_data]
    
    return (training_data, validation_data, test_data)

class warstwa():
    def __init__(self, rozmiar_poprzedniej_warstwy, rozmiar):
        self.rozmiar_poprzednej = rozmiar_poprzedniej_warstwy
        self.rozmiar = rozmiar
        self.bias = cp.random.randn(rozmiar)
        self.wagi = cp.random.randn(rozmiar, rozmiar_poprzedniej_warstwy)
        self.z = cp.zeros(rozmiar)
        self.output = cp.zeros(rozmiar)
        self.learning_rate = 0.03
        self.wejscie = None
        self.blad = None
        self.batch = 32

    def sigmoid(self, x):
        return 1.0 / (1.0 + cp.exp(-x))
    
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1.0 - sig)


    def feedforward(self, wejscie):
        self.wejscie = cp.array(wejscie)
        self.z = cp.dot(self.wagi, self.wejscie) + self.bias
        self.output = self.sigmoid(self.z)
        return self.output
        
    
    def backpropagation(self, blad):
        gradient = blad * self.sigmoid_derivative(self.z)
        delta_wagi = cp.dot(gradient, self.wejscie.T) / self.batch_size
        delta_bias = cp.sum(gradient, axis=1, keepdims=True) / self.batch_size
        self.wagi -= self.learning_rate * delta_wagi
        self.bias -= self.learning_rate * delta_bias
        self.blad = cp.dot(self.wagi.T, gradient)
        return self.blad


class siec():
    def __init__(self, siec_wymiary, dane, validation_data):
        self.validation_data = validation_data
        self.dane_z_wynikami = dane
        self.siec_wymiary = siec_wymiary
        self.siec = []
        for i in range(1, len(siec_wymiary)):
            self.siec.append(warstwa(siec_wymiary[i-1], siec_wymiary[i]))

    def feedforward(self, dane):
        for warstwa in self.siec:
            output = warstwa.feedforward(dane)
        return output

    def backpropagation(self, dane):
        blad = self.siec[-1].output - dane[1].flatten()
        for warstwa in self.siec[::-1]:
            blad = warstwa.backpropagation(blad)
   
    def train(self, epochs=5):
        for epoch in range(epochs):
            cp.random.shuffle(self.dane_z_wynikami)
            for i, dane in enumerate(self.dane_z_wynikami):
                self.feedforward(dane[0])
                self.backpropagation(dane)
                if (i + 1) % 1000 == 0:
                    print(f"Epoch {epoch + 1}, Sample {i + 1} trained.")
            accuracy = self.evaluate(self.validation_data)
            print(f"Epoch {epoch + 1} complete. Validation accuracy: {accuracy} / {len(self.validation_data)}")
    
    def evaluate(self, test_data):
        correct = 0
        for x, y in test_data:
            output = self.feedforward(x)
            predicted = cp.argmax(output)
            actual = cp.argmax(y) if isinstance(y, cp.ndarray) else int(y)
            correct += int(predicted == actual)
        return 


training_data, validation_data, test_data = load_data_wrapper_cupy()


dane_wejsciowe = 784
dane_wyjscowe = 10

siec1 = siec([dane_wejsciowe, 12, 12, dane_wyjscowe], training_data, validation_data)
siec1.train()