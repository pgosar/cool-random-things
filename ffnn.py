"""
A feed forward neural network built from scratch
"""

from random import seed, random
from math import exp


def initialize_network(num_inputs, num_hidden, num_outputs):
    network = []
    hidden_layer = []
    for _ in range(num_hidden):
        weights = {'weights': [random() for _ in range(num_inputs + 1)]}
        hidden_layer.append(weights)
    network.append(hidden_layer)

    output_layer = []
    for _ in range(num_outputs):
        weights = {'weights': [random() for _ in range(num_hidden + 1)]}
        output_layer.append(weights)
    network.append(output_layer)

    return network


def activate(weights, inputs):
    # bias node
    activation = weights[-1]
    # fully connected neural network, therefore activation must be summation
    # of all nodes in previous layer
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def forward_propagate(network, inputs):
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            # sigmoid activation function
            neuron['output'] = 1.0 / (1.0 + exp(-activation))
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        # everything besides output layer
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                # next layer closest to output
                for neuron in network[i + 1]:
                    # weighted error
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            # output layer error is just actual - expected
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        # calculate delta with sigmoid derivative
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * \
                neuron['output'] * (1.0 - neuron['output'])


def update_weights(network, row, learn_rate):
    for i in range(len(network)):
        # exclude target output
        inputs = row[:-1]
        # exclude input layer
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                # new weight = old weight - learn rate * error * input
                neuron['weights'][j] -= learn_rate * \
                    neuron['delta'] * inputs[j]
            # bias - input is 1
            neuron['weights'][-1] -= learn_rate * neuron['delta']


def train_network(network, train, learn_rate, num_epochs, num_outputs):
    for epoch in range(num_epochs):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for _ in range(num_outputs)]
            expected[row[-1]] = 1
            # SSE - sum squared error
            sum_error += sum([(expected[i]-outputs[i]) **
                             2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, learn_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' %
              (epoch, learn_rate, sum_error))


def predict(network, row):
    outputs = forward_propagate(network, row)
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
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)
for row in dataset:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))
