import math

from random import seed
from random import randrange
from random import random

from csv import reader


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def derivative(output):
    return output * (1.0 - output)


class Backpropagation(object):
    def __init__(
            self,
            learning_rate,
            num_iterations,
            hidden_layers,
            num_folds
        ):

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.hidden_layers = hidden_layers
        self.num_folds = num_folds

    def create_network(self, n_inputs, n_outputs):
        self.network = []
        hidden_layer = [{'weights':[
            random() for i in range(
                n_inputs + 1)]} for i in range(self.hidden_layers)
        ]
        self.network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(
            self.hidden_layers + 1)]} for i in range(n_outputs)]
        self.network.append(output_layer)

    def cross_validation_split(self, dataset):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / self.num_folds)
        for _ in range(self.num_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split


    def back_propagation(self, train, test):
        n_inputs = len(train[0]) - 1
        n_outputs = len(set([row[-1] for row in train]))
        self.create_network(n_inputs, n_outputs)

        self.train_network(train, n_outputs)
        predictions = list()
        for row in test:
            prediction = self.predict(row)
            # print('prediction', prediction)
            predictions.append(prediction)
        return(predictions)


    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))


    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                W = self.evaluate(neuron['weights'], inputs)
                neuron['output'] = sigmoid(W)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def evaluate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation


    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i != len(self.network) - 1:
                for j, _ in enumerate(layer):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j, _ in enumerate(layer):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j, _ in enumerate(layer):
                neuron = layer[j]
                neuron['delta'] = errors[j] * derivative(neuron['output'])


    def train_network(self, train, n_outputs):
        print_step = self.num_iterations / 10
        for i in range(self.num_iterations):
            if not i % print_step:
                print('iter %d' % i)
            for row in train:
                self.forward_propagate(row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                self.backward_propagate_error(expected)
                self.update_weights(row)

    def update_weights(self, row):
        for i, _ in enumerate(self.network):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j, _ in enumerate(inputs):
                    neuron['weights'][j] += self.learning_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += self.learning_rate * neuron['delta']

    def get_accuracy(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def run(self, dataset):
        folds = self.cross_validation_split(dataset)
        scores = []
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = []
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = self.back_propagation(train_set, test_set)
            actual = [row[-1] for row in fold]
            accuracy = self.get_accuracy(actual, predicted)
            scores.append(accuracy)
        print('accuracy', sum(scores) / float(len(scores)))


if __name__ == '__main__':
    learning_rate = 0.1
    num_iterations = 500
    hidden_layers = 6
    num_folds = 2
    model = Backpropagation(
        learning_rate, num_iterations, hidden_layers, num_folds)

    filename = 'seeds_dataset.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)

    str_column_to_int(dataset, len(dataset[0])-1)

    # minmax = dataset_minmax(dataset)
    # normalize_dataset(dataset, minmax)

    print('dataset', dataset)

    model.run(dataset)
    print(model.predict(dataset[]))
    # print(len(dataset[0]))