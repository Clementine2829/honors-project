from my_models.NeuralNetworkClass.layer import Layer
import numpy as np


# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, x_size, y_size):
        self.bias = np.random.rand(1, y_size) - 0.5
        self.weights = np.random.rand(x_size, y_size) - 0.5

    # returns output for a given input
    def forward(self, data_input):
        self.input = data_input
        # print("input: " + str(self.input) + " \nweight: " + str(self.weights) + " \nbias: " + str(self.bias))
        return np.dot(self.input, self.weights) + self.bias

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward(self, err_output, learning_rate):
        err_weights = np.dot(self.input.T, err_output)
        err_input = np.dot(err_output, self.weights.T)

        # update parameters
        self.bias -= learning_rate * err_output
        self.weights -= learning_rate * err_weights
        return err_input
