
class neuron():
    def __init__(self, rozmiar_poprzedniej_warstwy):
        self.rozmiar_poprzednej = rozmiar_poprzedniej_warstwy
        self.bias = [np.random.randn()]
        self.wagi = [np.random.randn(x) for x in range(rozmiar_poprzedniej_warstwy)]
        self.z = 0
        self.output = 0

        # TODO
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.sigmoid_derivative = lambda x: self.sigmoid(x) * (1 - self.sigmoid(x))

    def feedforward(self, wejscie):
        suma = np.dot(self.wagi, wejscie) + self.bias
        self.z = suma
        return self.sigmoid(suma)

class siec():
    def __init__(self, siec_wymiary):
        self.siec = siec

        