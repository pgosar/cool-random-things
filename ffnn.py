"""
A feed forward neural network built from scratch
"""

from random import seed, random
from math import exp


from random import seed, random
from math import exp


class Neuron:
    def __init__(self, weights):
        self.weights = weights
        self.output = None
        self.delta = None


class Network:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.layers = []
        self.initialize_network(num_inputs, num_hidden, num_outputs)

    def initialize_network(self, num_inputs, num_hidden, num_outputs):
        hidden_layer = [Neuron([random() for _ in range(num_inputs + 1)])
                        for _ in range(num_hidden)]
        self.layers.append(hidden_layer)

        output_layer = [Neuron([random() for _ in range(num_hidden + 1)])
                        for _ in range(num_outputs)]
        self.layers.append(output_layer)

    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += inputs[i] * weights[i]
        return activation

    def forward_propagate(self, inputs):
        for layer in self.layers:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron.weights, inputs)
                neuron.output = 1.0 / (1.0 + exp(-activation))
                new_inputs.append(neuron.output)
            inputs = new_inputs
        return inputs

    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            err = []
            if i == len(self.layers) - 1:
                for j, neuron in enumerate(layer):
                    err.append(neuron.output - expected[j])
            else:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.layers[i + 1]:
                        error += neuron.delta * neuron.weights[j]
                    err.append(error)
            for j, neuron in enumerate(layer):
                neuron.delta = err[j] * neuron.output * (1.0 - neuron.output)

    def update_weights(self, row, learn_rate):
        inputs = row[:-1]
        for i in range(len(self.layers)):
            if i != 0:
                inputs = [neuron.output for neuron in self.layers[i - 1]]
            for neuron in self.layers[i]:
                for j in range(len(inputs)):
                    neuron.weights[j] -= learn_rate * neuron.delta * inputs[j]
                neuron.weights[-1] -= learn_rate * neuron.delta

    def train(self, train_data, learn_rate, num_epochs, num_outputs):
        for epoch in range(num_epochs):
            sum_error = 0
            for row in train_data:
                outputs = self.forward_propagate(row)
                expected = [0 for _ in range(num_outputs)]
                expected[row[-1]] = 1
                sum_error += sum([(expected[i] - outputs[i])
                                 ** 2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row, learn_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' %
                  (epoch, learn_rate, sum_error))

    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))


seed(1)
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = Network(n_inputs, 2, n_outputs)
network.train(dataset, 0.5, 20, n_outputs)
for row in dataset:
    prediction = network.predict(row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))
