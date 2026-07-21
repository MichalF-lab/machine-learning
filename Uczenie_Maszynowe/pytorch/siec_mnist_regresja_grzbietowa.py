# -*- coding: utf-8 -*-
# %load mnist_loader.py
"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
import math

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    """
    try:
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            # Uzycie 'latin1' jest konieczne dla starszych plikow pickle Pythona 2
            training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    except FileNotFoundError:
        print("Blad: Plik 'mnist.pkl.gz' nie zostal znaleziony. Prosze go pobrac i umiescic w katalogu roboczym.")
        # Zwracanie pustych danych, aby umozliwic kontynuacje, chociaz program sie zalamie
        return ((np.array([]), np.array([])), (np.array([]), np.array([])), (np.array([]), np.array([])))
        
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Konwersja danych na format (wejscia, wektor_wyjscia) dla treningu
    oraz (wejscia, etykieta) dla walidacji/testu.
    """
    tr_d, va_d, te_d = load_data()
    
    training_inputs = [x.reshape((784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [x.reshape((784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [x.reshape((784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


# --- KOD SIECI NEURONOWEJ ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class Network(nn.Module):
    def __init__(self, sizes):
        super(Network, self).__init__()
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
            
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=1.0 / math.sqrt(sizes[0]))
            nn.init.normal_(layer.bias, mean=0.0, std=1.0 / math.sqrt(sizes[0]))
    
    def forward(self, x):
        x = x.view(-1) 
        for layer in self.layers:
            if layer == self.layers[-1]:
                x = layer(x)
            else:
                x = torch.sigmoid(layer(x))
        return x

    def ridge_fit(self, training_data_list, alpha=0.2, learning_rate=0.0005):
        
        X_rows = []
        Y_rows = []

        with torch.no_grad():
            for x, y in training_data_list:
                
                x_tensor = torch.tensor(x.ravel(), dtype=torch.float32).to(device)
                
                a = x_tensor
                for layer in self.layers[:-1]:
                    a = torch.sigmoid(layer(a)).to(device)
                
                X_rows.append(a.cpu().numpy())
                Y_rows.append(y.ravel())
        
        # Tworzenie macierzy X i Y na GPU
        X = torch.tensor(np.vstack(X_rows), dtype=torch.float32).to(device)
        Y = torch.tensor(np.vstack(Y_rows), dtype=torch.float32).to(device)
        
        # Dodanie bias term
        ones = torch.ones(X.shape[0], 1, device=device)
        X_with_bias = torch.cat([ones, X], dim=1)
        
        XtX = torch.matmul(X_with_bias.T, X_with_bias)
        I = torch.eye(X_with_bias.shape[1], device=device)
        XtY = torch.matmul(X_with_bias.T, Y)
        
        # Rozwiazanie za pomoca stabilnej metody linalg.solve (dziala na GPU)
        theta = torch.linalg.solve(XtX + alpha * I, XtY)
        
        # Aktualizacja ostatniej warstwy
        with torch.no_grad():
            self.layers[-1].bias.data = theta[0, :].reshape(-1)
            self.layers[-1].weight.data = theta[1:, :].T

    def evaluate(self, test_data):
        correct = 0
        total = len(test_data)
        
        with torch.no_grad():
            for x, y in test_data:
                
                x_tensor = torch.tensor(x.ravel(), dtype=torch.float32).to(device)
                
                output = self.forward(x_tensor)
                pred = torch.argmax(output).item()
                
                if pred == y:
                    correct += 1
        return correct


# --- Uruchomienie Kodu ---

training_data, validation_data, test_data = load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)


net2 = Network([784, 50, 50])
net2.to(device)

net2.ridge_fit(training_data, alpha=0.3, learning_rate=0.0009)

accuracy = net2.evaluate(validation_data)
print(f"Validation Accuracy: {accuracy} / {len(validation_data)}")