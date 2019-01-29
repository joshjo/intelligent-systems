import re
import numpy as np
import pandas as pd


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text.strip()


# if __name__ == '__main__':
data = pd.read_csv('spanish_emojis.csv')
observations = list(data['observations'])
words = [tokenizer(i).split(' ') for i in observations]
print(words[0])
