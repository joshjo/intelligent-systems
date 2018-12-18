# coding: utf-8
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
get_ipython().run_line_magic('pinfo', 'model.similarity')
model.wv.similarity
get_ipython().run_line_magic('pinfo', 'model.wv.similarity')
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
get_ipython().run_line_magic('pinfo', 'model.wv.similar_by_word')
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
get_ipython().run_line_magic('pinfo', 'model.train')
model.wv['writer later']
model.wv['writer']
model.wv['writer']
model.wv['writer years']
model.vector_size
model.sample
get_ipython().run_line_magic('pinfo', 'model.wv')
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
get_ipython().run_line_magic('save', 'current_session ~0/')
