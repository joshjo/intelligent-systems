
# Backpropagation for Sentiment Analysis

The following algorithm uses a Backpropagation with a SGD optimizer

### The IMDb Movie Review Dataset

In this section, we will train a simple logistic regression model to classify movie reviews from the 50k IMDb review dataset that has been collected by Maas et. al.

AL Maas, RE Daly, PT Pham, D Huang, AY Ng, and C Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Lin- guistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics

[Source: http://ai.stanford.edu/~amaas/data/sentiment/]

The dataset consists of 50,000 movie reviews from the original "train" and "test" subdirectories. The class labels are binary (1=positive and 0=negative) and contain 25,000 positive and 25,000 negative movie reviews, respectively. For simplicity, I assembled the reviews in a single CSV file.


```python
import re
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
```

### Preprocessing text data


```python
stop = stopwords.words('english')
porter = PorterStemmer()

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return text
```

Let's give it at try:


```python
tokenizer('This :) is a <a> test! :-)</br>')
```




    ['test', ':)', ':)']



### Import dataset and preparing using word2vec (Exercise 1)


```python
def featureVecMethod(words, model, num_features):
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0

    index2word_set = set(model.wv.index2word)

    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])

    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(reviews)))

        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1

    return reviewFeatureVecs
```


```python
df = pd.read_csv('shuffled_movie_data.csv')
```


```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>49995</th>
      <td>OK, lets start with the best. the building. al...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49996</th>
      <td>The British 'heritage film' industry is out of...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49997</th>
      <td>I don't even know where to begin on this one. ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49998</th>
      <td>Richard Tyler is a little boy who is scared of...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49999</th>
      <td>I waited long to watch this movie. Also becaus...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We need to tokenize each review. This will take approx 3 minutes.


```python
X = list(df['review'])
y = list(df['sentiment'])
xx = []
for i in X:
    xx.append(tokenizer(i))
```

Lets check some data



```python
xx[0]
```




    ['1974',
     'teenager',
     'martha',
     'moxley',
     'maggie',
     'grace',
     'moves',
     'high',
     'class',
     'area',
     'belle',
     'greenwich',
     'connecticut',
     'mischief',
     'night',
     'eve',
     'halloween',
     'murdered',
     'backyard',
     'house',
     'murder',
     'remained',
     'unsolved',
     'twenty',
     'two',
     'years',
     'later',
     'writer',
     'mark',
     'fuhrman',
     'christopher',
     'meloni',
     'former',
     'la',
     'detective',
     'fallen',
     'disgrace',
     'perjury',
     'j',
     'simpson',
     'trial',
     'moved',
     'idaho',
     'decides',
     'investigate',
     'case',
     'partner',
     'stephen',
     'weeks',
     'andrew',
     'mitchell',
     'purpose',
     'writing',
     'book',
     'locals',
     'squirm',
     'welcome',
     'support',
     'retired',
     'detective',
     'steve',
     'carroll',
     'robert',
     'forster',
     'charge',
     'investigation',
     '70',
     'discover',
     'criminal',
     'net',
     'power',
     'money',
     'cover',
     'murder',
     'murder',
     'greenwich',
     'good',
     'tv',
     'movie',
     'true',
     'story',
     'murder',
     'fifteen',
     'years',
     'old',
     'girl',
     'committed',
     'wealthy',
     'teenager',
     'whose',
     'mother',
     'kennedy',
     'powerful',
     'rich',
     'family',
     'used',
     'influence',
     'cover',
     'murder',
     'twenty',
     'years',
     'however',
     'snoopy',
     'detective',
     'convicted',
     'perjurer',
     'disgrace',
     'able',
     'disclose',
     'hideous',
     'crime',
     'committed',
     'screenplay',
     'shows',
     'investigation',
     'mark',
     'last',
     'days',
     'martha',
     'parallel',
     'lack',
     'emotion',
     'dramatization',
     'vote',
     'seven',
     'title',
     'brazil',
     'available']



Create the word2vec model and train with all the words


```python
model = Word2Vec(xx, size=100, window=5, min_count=1, workers=4)
```

Check some words and its distance to verify the model was trained correctly.


```python
model.wv.similar_by_word('paris')
```

    /home/josue/.virtualenvs/is/lib/python3.6/site-packages/gensim/matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int64 == np.dtype(int).type`.
      if np.issubdtype(vec.dtype, np.int):





    [('aime', 0.8182104825973511),
     ('je', 0.7898210883140564),
     ('italy', 0.7548259496688843),
     ('berlin', 0.753639817237854),
     ('france', 0.7418023347854614),
     ('london', 0.7360866665840149),
     ('rome', 0.7088476419448853),
     ('england', 0.7076780796051025),
     ('slovakia', 0.7063679695129395),
     ('san', 0.7044035792350769)]




```python
model.wv.similar_by_word('dog')
```




    [('cat', 0.776843786239624),
     ('puppy', 0.7679356932640076),
     ('freak', 0.7312381863594055),
     ('bite', 0.709902286529541),
     ('pet', 0.6959309577941895),
     ('bugs', 0.6937322616577148),
     ('rat', 0.6844397783279419),
     ('chicken', 0.664679765701294),
     ('monkey', 0.6636012196540833),
     ('dogs', 0.6596550941467285)]



Prepare data with word2vec model. This will take 15 or 20 minutes, depends on your computer.


```python
x_data = getAvgFeatureVecs(xx, model, 100)
```

    Review 0 of 50000


    /home/josue/.virtualenvs/is/lib/python3.6/site-packages/ipykernel_launcher.py:10: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      # Remove the CWD from sys.path while we load stuff.


    Review 1000 of 50000
    Review 2000 of 50000
    Review 3000 of 50000
    Review 4000 of 50000
    Review 5000 of 50000
    Review 6000 of 50000
    Review 7000 of 50000
    Review 8000 of 50000
    Review 9000 of 50000
    Review 10000 of 50000
    Review 11000 of 50000
    Review 12000 of 50000
    Review 13000 of 50000
    Review 14000 of 50000
    Review 15000 of 50000
    Review 16000 of 50000
    Review 17000 of 50000
    Review 18000 of 50000
    Review 19000 of 50000
    Review 20000 of 50000
    Review 21000 of 50000
    Review 22000 of 50000
    Review 23000 of 50000
    Review 24000 of 50000
    Review 25000 of 50000
    Review 26000 of 50000
    Review 27000 of 50000
    Review 28000 of 50000
    Review 29000 of 50000
    Review 30000 of 50000
    Review 31000 of 50000
    Review 32000 of 50000
    Review 33000 of 50000
    Review 34000 of 50000
    Review 35000 of 50000
    Review 36000 of 50000
    Review 37000 of 50000
    Review 38000 of 50000
    Review 39000 of 50000
    Review 40000 of 50000
    Review 41000 of 50000
    Review 42000 of 50000
    Review 43000 of 50000
    Review 44000 of 50000
    Review 45000 of 50000
    Review 46000 of 50000
    Review 47000 of 50000
    Review 48000 of 50000
    Review 49000 of 50000



```python
x_data[0]
```




    array([ 0.07630794, -0.01097148, -0.12956089,  0.27931604, -0.20665306,
           -0.21279204,  0.12511088, -0.02736017,  0.44785362,  0.28132117,
            0.31679863, -0.26683202, -0.09816159, -0.07439731, -0.2067431 ,
            0.04911201, -0.16600397, -0.05123523, -0.02551782, -0.20975916,
            0.56280106, -0.5263951 , -0.14751084, -0.05870283, -0.35502002,
           -0.22698343, -0.65806246,  0.29519364,  0.39090404, -0.32554492,
            0.45354035,  0.24128316,  0.04480733, -0.22661589, -0.11324843,
            0.00905388,  0.25195992,  0.5604066 , -0.07369114, -0.29293683,
            0.23193975, -0.14690454,  0.12431493, -0.15815154, -0.09562442,
           -0.09932413,  0.29504433,  0.04829625,  0.20742415, -0.0076751 ,
           -0.4864169 ,  0.28021845, -0.4128799 , -0.03994881, -0.08676451,
           -0.0482845 , -0.257897  ,  0.3056662 , -0.30582327,  0.43214902,
           -0.06880512,  0.6272548 ,  0.21559963, -0.11575905, -0.30783722,
           -0.5108224 , -0.23295109,  0.09969001, -0.13333625,  0.30695784,
            0.27372244, -0.18227015, -0.14265977,  0.25540152, -0.23985222,
            0.5484486 , -0.16056497, -0.31099397, -0.5217287 , -0.35832378,
           -0.32457694,  0.12634073, -0.35488126, -0.00155702,  0.00526529,
            0.10731545, -0.22469126, -0.39370438,  0.0559814 ,  0.1661358 ,
           -0.07472399, -0.46182725, -0.02262228,  0.3908494 ,  0.3356843 ,
            0.0116782 ,  0.20559731, -0.21520431, -0.04848151,  0.05598722],
          dtype=float32)




```python
x_data = [list(i) for i in x_data]
```


```python
from copy import deepcopy
```


```python
dataset = deepcopy(x_data)
```


```python
for x, yy in zip(dataset, y):
    x.append(yy)
```


```python
len(dataset[0])
```




    101



## Implement a backpropagation algorithm using SGD (Exercise 2)

The algorithm is implemented in the file `backpropagtion.py`


```python
from backpropagation import *
```

Initialize the parameters


```python
learning_rate = 0.1
num_iterations = 10
hidden_layers = 6
num_folds = 2
```


```python
bpmodel = Backpropagation(
        learning_rate, num_iterations, hidden_layers, num_folds)
```

Train data using cross validation. The accyracy approximately should be higher than 80%


```python
bpmodel.run(dataset)
```

    iter 0
    iter 1
    iter 2
    iter 3
    iter 4
    iter 5
    iter 6
    iter 7
    iter 8
    iter 9
    iter 0
    iter 1
    iter 2
    iter 3
    iter 4
    iter 5
    iter 6
    iter 7
    iter 8
    iter 9
    accuracy 86.402



```python
example = tokenizer('I loved this movie')
X = featureVecMethod(example, model, 100)
bpmodel.predict(X)
```

    /home/josue/.virtualenvs/is/lib/python3.6/site-packages/ipykernel_launcher.py:10: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      # Remove the CWD from sys.path while we load stuff.





    1




```python
example = tokenizer('I did not like this movie')
X = featureVecMethod(example, model, 100)
bpmodel.predict(X)
```

    /home/josue/.virtualenvs/is/lib/python3.6/site-packages/ipykernel_launcher.py:10: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      # Remove the CWD from sys.path while we load stuff.





    0




```python

```
