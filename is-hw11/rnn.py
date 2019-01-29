import re
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text.strip()

# if __name__ == '__main__':
data = pd.read_csv('spanish_emojis.csv')
observations = list(data['observations'])
words = [tokenizer(i) for i in observations]

tokenizer = Tokenizer(nb_words=2500, lower=True,split=' ')
tokenizer.fit_on_texts(words)
X = tokenizer.texts_to_sequences(words)
X = pad_sequences(X)
