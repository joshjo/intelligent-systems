from prueba import *
df
list(df)
list(df['review'])
X = list(df['review'])
X
len(X)
y = list(df['sentiment'])
len(y)
X[0]
tokenizer(X[0])
model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
from gensim.test.utils import common_texts, get_tmpfile
model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
X
xx = []
for i in X:
    xx.append(tokenizer(i))
xx
len(xx)
xx[0]
xx[1]
model
model.train(xx, total_examples=1, epochs=1)
model.wv
model.wv['hello']
model.wv.vocabulary
model.wv.vocab
model.wv['human']
model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
model.train(xx[0], total_examples=1, epochs=1)
model.train(xx[0], epochs=1)
model.train(xx[0], total_examples=2, epochs=1)
model.train(xx[0], total_examples=3, epochs=1)
model.wv.vectors
len(model.wv.vectors)
xx[0]
len(xx[0])
model.train(xx[0], total_examples=3, epochs=1)
model.wv
model.train(xx[0], total_examples=3, epochs=1)
len(model.wv.vectors)
model.wv
model.wv[0]
model.wv.word_vec
model.wv.word_vec()
model.wv.vocab
model.wv.vocab['human']
xx[0]
model.wv.vocab
model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
model.wv.vocab
common_texts
model = Word2Vec(xx, size=100, window=5, min_count=1, workers=4)
model.wv.vocab
model.train(xx[0], total_examples=1, epochs=1)
model.train(xx[0], total_examples=1, epochs=1)
model.train(xx[0], total_examples=1, epochs=1)
model.train(xx[0], total_examples=1, epochs=1)
model.train(xx[1], total_examples=1, epochs=1)
model.train(xx[1], total_examples=2, epochs=1)
model.train(xx[1], total_examples=20, epochs=1)
model.train(xx[1], total_examples=20, epochs=2)
model.train(xx[1], total_examples=20, epochs=3)
model.train(xx[1], total_examples=20, epochs=4)
model.train(xx[1], total_examples=20, epochs=5)
model.train(xx[1], total_examples=20, epochs=100)
model.train(xx[1], total_examples=20, epochs=2)
model.similarity?
model.wv.similarity
model.wv.similarity?
model.wv.similarity('teenager', 'martha')
model.wv.distance('teenager')
model.wv.distances('teenager')
d= model.wv.distances('teenager')
d
d[0]
len(d)
d= model.wv.distances('france', 'spain')
d= model.wv.distance('france', 'spain')
d= model.wv.distances('france')
model.wv.distances('france')
model.wv.distance('france')
model.wv.similar_by_word
model.wv.similar_by_word?
model.wv.similar_by_word('teenager')
model.wv.similar_by_word('paris')
model.wv.similar_by_word('dog')
model.wv.similar_by_word('dog', 15)
model.wv.similar_by_word('dog')
model.wv.similar_by_word('sol')
model.wv.similar_by_word('josue')
model.wv.similar_by_word('wallet')
model.wv.similar_by_word('bottle')
model.wv.similar_by_word('fireplace')
xx
xx[0]
model.wv.similar_by_word('seven')
model.train(xx[2])
xx
xx[2]
model.train(xx[2], total_examples=20, epochs=2)
model.train(xx[1], total_examples=20, epochs=2)
model.train(xx[2], total_examples=20, epochs=2)
model.train(xx[2], total_examples=2, epochs=2)
model.train(xx[2], total_examples=1223, epochs=2)
model.train(xx[2], total_examples=1223, epochs=3)
model.train(xx[2], total_examples=1223, epochs=1)
model.train?
model.wv['writer later']
model.wv['writer']
model.wv['writer']
model.wv['writer years']
model.vector_size
model.sample
model.wv?
dict(model.wv)
len(model.wv['sol'])
len(model.wv['paris'])
index2word_set = set(model.wv.index2word)
index2word_set
len(index2word_set)
index2word_set['twinkies']
# Function to average all word vectors in a paragraph
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec
featureVecMethod(xx[0], model, 100)
# Function for calculating the average feature vector
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1
        
    return reviewFeatureVecs
X
%save current_session ~0/
%history -f history.py
xx
x_data =getAvgFeatureVecs(xx, model, 100)
x_data
list(x_data)
x_data = [list(i) for i in x_data]
x_data
x_data[0]
y
type(y)
len(x_data)
data_set = []
x_data
len(x_data[0])
from copy import deepcopy
data_set = deepcopy(x_data)
for x, yy in zip(data_set, y):
    x.push_back(yy)
for x, yy in zip(data_set, y):
    x.append(yy)
len(data_set[0])
from backpropagation import *
n_folds = 2
    l_rate = 0.3
    n_epoch = 500
    n_hidden = 5
    n_folds = 2
    l_rate = 0.3
    n_epoch = 500
    n_hidden = 5
dataset = data_set
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
from backpropagation import *
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        value = row[column].strip()
        if value:
            row[column] = float(value)
        else:
            row[column] = 1


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        print('epoch %d' % epoch)
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    print('initialized network')
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)
n_folds = 2
    l_rate = 0.3
    n_epoch = 100
    n_hidden = 5
    n_folds = 2
    l_rate = 0.3
    n_epoch = 100
    n_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
n_epoch = 10
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
scores
print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
n_epoch = 5
scores5 = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
scores5
n_epoch = 1
scores1 = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
scores1
n_hidden = 3
l_rate = 0.01
scores1 = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
scores1
l_rate = 0.3
scores1 = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
scores1
scores5 = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
tokenizer('I loved this movie')
h = tokenizer('I loved this movie')
getAvgFeatureVecs(h)
getAvgFeatureVecs(h, model, 100)
h = tokenizer('I loved this movie sadasd')
h
getAvgFeatureVecs(h, model, 100)
h
getAvgFeatureVecs([h], model, 100)
getAvgFeatureVecs([h[:2]], model, 100)
h[:2]
history -f history.py
