
# Hidden Markov Network

We are going to use the data from https://www.clips.uantwerpen.be/conll2002/ner/

## Import the data


```python
f = open('esp.train', encoding='latin-1')
lines = f.readlines()
f.close()
```

We are going to get the states and the start PI probability for each tag


```python
from collections import Counter

states = []

states_occurrences = Counter()

data = []
words_list = []

for line in lines:
    words = line.split()
    if len(words) != 2:
        continue
    if words[0] not in words_list:
        words_list.append(words[0])
    data.append(words)
    states_occurrences[words[1]] += 1

len_data = len(data)

start_probability = {
    k: (v / len_data) for k, v in states_occurrences.items()
}
```

Now we are going to get the transition probability


```python
transition_probability = []

states_list = states_occurrences.keys()

for state in states_list:
    eachProb = []

    for each_state in states_list:
        count_transition = 0
        count_state = 0
        for x in range(len(data) - 1):
            first = data[x]
            second = data[x + 1]
            if first[1] == state:
                count_state += 1
                if second[1] == each_state:
                    count_transition += 1
        eachProb.append(count_transition / count_state)
    transition_probability.append(eachProb)
```


```python
transition_probability
```




    [[0.005292082230816202,
      0.7748829635660492,
      0.0,
      0.002646041115408101,
      0.0,
      0.0,
      0.0,
      0.21717891308772644,
      0.0],
     [0.020770182693095433,
      0.9199030696061987,
      0.031791271952707624,
      0.01820463178954721,
      0.0,
      0.009330843958451011,
      0.0,
      0.0,
      0.0],
     [0.0040595399188092015,
      0.6863328822733423,
      0.00013531799729364006,
      0.005953991880920162,
      0.0,
      0.00013531799729364006,
      0.303382949932341,
      0.0,
      0.0],
     [0.0006942837306179125,
      0.333256190696598,
      0.00023142791020597085,
      0.00023142791020597085,
      0.6655866697523721,
      0.0,
      0.0,
      0.0,
      0.0],
     [0.0058929028952088135,
      0.7294388931591084,
      0.0,
      0.0015372790161414297,
      0.2631309249295414,
      0.0,
      0.0,
      0.0,
      0.0],
     [0.0027611596870685687,
      0.41187298665439487,
      0.004601932811780948,
      0.0032213529682466636,
      0.0,
      0.0,
      0.0,
      0.0,
      0.577542567878509],
     [0.00040064102564102563,
      0.4439102564102564,
      0.0006009615384615385,
      0.004006410256410256,
      0.0,
      0.00020032051282051281,
      0.5508814102564102,
      0.0,
      0.0],
     [0.0010576414595452142,
      0.5631940772078265,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.4357482813326282,
      0.0],
     [0.0009339975093399751,
      0.3844956413449564,
      0.0006226650062266501,
      0.0024906600249066002,
      0.0,
      0.0021793275217932753,
      0.0,
      0.0,
      0.6092777085927771]]



We want to check probabilities of each tag.


```python
start_probability
```




    {'B-LOC': 0.01855958294769847,
     'O': 0.8761120450295601,
     'B-ORG': 0.02791681619855316,
     'B-PER': 0.016323215533687173,
     'I-PER': 0.014744158812307576,
     'B-MISC': 0.008208828362578623,
     'I-ORG': 0.018858017112743895,
     'I-LOC': 0.007143531722796215,
     'I-MISC': 0.012133804280074798}



Now we are going to calculate the emission probability


```python
emission_probability = []


for state, occurrences in states_occurrences.items():
    emission_probability_of_tag = []
    for word in words_list:
        count = 0
        for word_data, tag_data in data:
            if word == word_data and tag_data == state:
                count += 1
        try:
            emission_probability_of_tag.append(count / occurrences)
        except ZeroDivisionError:
            continue
    emission_probability.append(emission_probability_of_tag)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-7-36ff90842cfc> in <module>
          7         count = 0
          8         for word_data, tag_data in data:
    ----> 9             if word == word_data and tag_data == state:
         10                 count += 1
         11         try:


    KeyboardInterrupt: 



```python
def viterbi(y, A, B, Pi=None):
    # Cardinality of the state space
    K = A.shape[0]

    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2
```


```python

```
