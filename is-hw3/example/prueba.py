import re
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


df = pd.read_csv('shuffled_movie_data.csv')


X = df['review']
y = df['sentiment']

print(X[0])
