# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward(self, input):
        ...

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward(self, err_output, learning_rate):
        ...
