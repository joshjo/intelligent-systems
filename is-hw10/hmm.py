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

<<<<<<< HEAD
print('start_probability', start_probability)


# # find transition probability from tag Ti ti tag Tj
# transition_probability = []


# for state in states:
#     eachProb = []
#     for eachState in states:
#         count_transition = 0
#         count_state = 0
#         for line in data:
#             line = line.split()
#             for x in range(len(line) - 1):
#                 first = line[x].split('|')
#                 second = line[x+1].split('|')
#                 if first[1] == state:
#                     count_state += 1
#                     if second[1] == eachState:
#                         count_transition += 1

#         prob = count_transition / count_state
#         eachProb.append(prob)

#     transition_probability.append(eachProb)

# # print(states)
# # for state, prob in zip(states, transition_probability):
# #     print(state, prob)


# # find emission probability
# emission_probability = []

# # find list of all words
# words_list = []
# for line in data:
#     line = line.split()
#     for word in line:
#         word = word.split('|')
#         if word[0] not in words_list:
#             words_list.append(word[0])

# # find total occurrences of each tag
# occurances_of_tags = []
# for state in states:
#     count = 0
#     for line in data:
#         line = line.split()
#         for word in line:
#             word = word.split('|')
#             if word[1] == state:
#                 count += 1
#     occurances_of_tags.append(count)
# print(occurances_of_tags)

# # calculate emission probability
# for tag, occurances in zip(states, occurances_of_tags):
#     emission_probability_of_tag = []
#     for word in words_list:
#         count = 0
#         for line in data:
#             line = line.split()
#             for eachWord in line:
#                 eachWord = eachWord.split('|')
#                 if eachWord[0] == word and eachWord[1] == tag:
#                     count += 1
#         try:
#             emission_probability_of_tag.append(count / occurances)
#         except ZeroDivisionError:
#             continue
#     emission_probability.append(emission_probability_of_tag)

=======
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
>>>>>>> 345a96ea2ca0885f4c5b1d5e58d897f55a195fee
