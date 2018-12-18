
# Recurrent Neural Networks


The following algorithm uses a Recurrent Neural Network using keras

### The IMDb Movie Review Dataset

In this section, we will train a simple logistic regression model to classify movie reviews from the 50k IMDb review dataset that has been collected by Maas et. al.

AL Maas, RE Daly, PT Pham, D Huang, AY Ng, and C Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Lin- guistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics

[Source: http://ai.stanford.edu/~amaas/data/sentiment/]

The dataset consists of 50,000 movie reviews from the original "train" and "test" subdirectories. The class labels are binary (1=positive and 0=negative) and contain 25,000 positive and 25,000 negative movie reviews, respectively. For simplicity, I assembled the reviews in a single CSV file.


## 1. Preprocess Data

We are going to import the data and tokenize using the Tokeinzer from keras


```python
import re
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

stop = stopwords.words('english')
porter = PorterStemmer()


def xtokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return text
```

Read data using `pandas`


```python
df = pd.read_csv('shuffled_movie_data.csv')

df = df[:30000]

X = df['review']
y = df['sentiment']
```

Preprocessing data with the tokenizer. This could take more than 3 minutes


```python
xx = np.array([tokenizer(i) for i in X])
```


```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
```


```python
tokenizer = Tokenizer(nb_words=2500, lower=True,split=' ')
tokenizer.fit_on_texts(xx)
X = tokenizer.texts_to_sequences(xx)
X = pad_sequences(X)
```

    /home/josue/.virtualenvs/is/lib/python3.6/site-packages/keras_preprocessing/text.py:177: UserWarning: The `nb_words` argument in `Tokenizer` has been renamed `num_words`.
      warnings.warn('The `nb_words` argument in `Tokenizer` '


Now we are going to create the network with `keras`


```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split
```


```python
embed_dim = 128
lstm_out = 200
batch_size = 32

model = Sequential()
model.add(Embedding(10000, embed_dim, input_length = X.shape[1], dropout = 0.2))
#model.add(Bidirectional(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2)))
model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
```

    /home/josue/.virtualenvs/is/lib/python3.6/site-packages/ipykernel_launcher.py:6: UserWarning: The `dropout` argument is no longer support in `Embedding`. You can apply a `keras.layers.SpatialDropout1D` layer right after the `Embedding` layer to get the same behavior.
      
    /home/josue/.virtualenvs/is/lib/python3.6/site-packages/ipykernel_launcher.py:8: UserWarning: Update your `LSTM` call to the Keras 2 API: `LSTM(200, dropout=0.2, recurrent_dropout=0.2)`
      


    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 772, 128)          1280000   
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 200)               263200    
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 402       
    =================================================================
    Total params: 1,543,602
    Trainable params: 1,543,602
    Non-trainable params: 0
    _________________________________________________________________
    None



```python
Y = pd.get_dummies(y).values
X_train, X_valid, ç, Y_valid = train_test_split(X, Y, test_size = 0.20, random_state = 36)
```

Now we are going to train our neural network. This can take more than 2 hours.


```python
f= model.fit(X_train, Y_train, batch_size =batch_size, nb_epoch = 10,  verbose = 5)
```

    /home/josue/.virtualenvs/is/lib/python3.6/site-packages/ipykernel_launcher.py:1: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
      """Entry point for launching an IPython kernel.


    Epoch 1/10
    Epoch 2/10
    Epoch 3/10
    Epoch 4/10
    Epoch 5/10
    Epoch 6/10
    Epoch 7/10
    Epoch 8/10
    Epoch 9/10
    Epoch 10/10


We can check the accuracy throug the iterations. It can reach up to 96%.


```python
f.history
```




    {'loss': [0.40947422365347547,
      0.3014738180736701,
      0.26317668383320175,
      0.22936233830451966,
      0.20843031751612823,
      0.18849573659648497,
      0.1595802935535709,
      0.13925711729253332,
      0.1167232109811157,
      0.10206902485589187],
     'acc': [0.8195,
      0.8785,
      0.893625,
      0.9081666666666667,
      0.9183333333333333,
      0.92775,
      0.9399166666666666,
      0.947,
      0.957375,
      0.9632083333333333]}



Before we continue to test, we are goig to save our model


```python
model.save('rnn.h5')
```

Now we  are going to test our `test_valid`


```python
y_results = model.predict_classes(X_valid)
```


```python
y_results
```




    array([1, 0, 1, ..., 1, 1, 1])




```python
corrects = [1 for valid, prediction in zip(Y_valid, y_results) if valid[1] == prediction]
```

Now we can check our test accuracy:


```python
print('%.2f%% accuracy' % (sum(corrects) / len(Y_valid) * 100))
```

    85.68% accuracy



```python

```
