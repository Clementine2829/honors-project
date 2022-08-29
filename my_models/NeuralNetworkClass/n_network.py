class MyNeuralNetwork:
    def __init__(self):
        self.loss = None
        self.loss_prime = None
        self.layers = []

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss_prime = loss_prime
        self.loss = loss

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    # get accuracy
    def get_accuracy(self, predicted, truth):
        accuracy = 0
        length = len(predicted)
        x = []
        y = []
        for i in range(length):
            temp_arr = predicted[i].reshape(-1).tolist()

            arr = [0, 1]
            if temp_arr[0] >= temp_arr[1]:
                arr = [1, 0]

            if arr[0] == truth[i][0] and arr[1] == truth[i][1]:
                accuracy += 1
                x.append(1)
            else:
                x.append(0)

            if truth[i][0] == 1:
                y.append(1)
            else:
                y.append(0)

        return round((accuracy / length) * 100, 2), x, y

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    # print("output: " + str(output))
                    output = layer.forward(output)

                # compute loss (for display purpose only)
                # err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            # calculate average error on all samples
            # err /= samples
            # print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
