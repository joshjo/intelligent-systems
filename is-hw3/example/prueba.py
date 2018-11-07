import re
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


stop = stopwords.words('english')
porter = PorterStemmer()

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return text


def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    for _ in range(size):
        text, label = next(doc_stream)
        docs.append(text)
        y.append(label)
    return docs, y


df = pd.read_csv('shuffled_movie_data.csv')


X = df['review']
y = df['sentiment']


if __name__ == '__main__':
    print(tokenizer('This :) is a <a> test! :-)</br>'))
    print(X[2])
