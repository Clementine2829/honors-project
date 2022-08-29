from my_models.NeuralNetworkClass.layer import Layer


# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, derevatives):
        self.activation = activation
        self.derevatives = derevatives

    # returns the activated input
    def forward(self, input_data):
        self.input = input_data
        return self.activation(self.input)

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward(self, err_output, learning_rate):
        derivative = self.derevatives(self.input)
        return derivative * err_output
