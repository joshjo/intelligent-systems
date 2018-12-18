from collections import Counter
import numpy as np


f = open('xxx.train', encoding='latin-1')
lines = f.readlines()
f.close()

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


print('emission_probability', emission_probability)
