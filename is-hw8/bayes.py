import math
import operator

from collections import defaultdict


def mean(numbers):
    n = len(numbers)
    return float(sum(numbers)) / n


def stdev(numbers):
    avg = mean(numbers)
    n = len(numbers)
    sum_diff = sum([pow(x - avg, 2) for x in numbers])
    return math.sqrt(float(sum_diff) / (n - 1))


def gauss(x, mean, stdev):
    """
    rename to gauss
    """

    exponent = math.exp(
        -(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


class NaiveBayes(object):
    def __init__(self):
        self.probabilities = {}
        self.prob_y = defaultdict(float)

    def train(self, X, Y):
        dim_X = len(X[0])
        size_data = len(X)
        factor_size = float(1) / size_data
        class_attrs = {i: [[] for j in range(dim_X)] for i in set(Y)}

        for x, y in zip(X, Y):
            self.prob_y[y] += factor_size
            for i, attr in enumerate(x):
                class_attrs[y][i].append(attr)

        for k, attrs in class_attrs.items():
            probability = []
            for i, attr in enumerate(attrs):
                probability.append({
                    'mean': mean(attr),
                    'stdev': stdev(attr),
                })
            self.probabilities[k] = probability

    def predict(self, X):
        if not self.probabilities:
            print("You should train the model before")
            return

        posteriors = {}
        evidence = 0
        for k, prob in self.probabilities.items():
            post = self.prob_y[k]
            for i, x in enumerate(X):
                mean, stdev = prob[i]['mean'], prob[i]['stdev']
                post *= gauss(x, mean, stdev)
            posteriors[k] = post
            evidence += post
        for k, v in posteriors.items():
            posteriors[k] = v / evidence
        return max(posteriors.items(), key=operator.itemgetter(1))[0]


if __name__ == '__main__':
    model = NaiveBayes()
    X = [[0, 5], [2, 3], [4, 1], [6, 2], [5, 2]]
    Y = [1, 1, 0, 0, 0]
    model.train(X, Y)
    print(
        model.predict([1, 7])

        # model.probabilities
        # model.get_mean_stdevs([[0, 5], [2, 3], [1, 4]], [1, 1, 0])
    )
    # model.train([1, 0.5])
    # print(model.probabilities)
    # print()
    # print('xxx')
