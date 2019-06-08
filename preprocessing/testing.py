import string
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import numpy as np
import string
import re

punctuations = [x for x in string.punctuation]

def remove_punctuation(sentence):
    words = sentence.split()
    new_sentence_ = ""
    for word in words:
        new_word = ""
        for x in word:
            if not(x in punctuations):
                new_word = new_word + x
        new_sentence_ = new_sentence_ + " " + new_word

    return new_sentence_.strip()

def text_preprocess(descriptions):
    processed_desc = []
    stemmer = SnowballStemmer("russian")

    for description in descriptions:
        print(description)
        description = description.lower()
        print(description)
        description = remove_punctuation(description)
        print(description)
        description = [stemmer.stem(word) for word in description.split()]
        print(description)
        print("================================")
        processed_desc.append(description)
    return


# train_info_path = "dataset/avito-duplicate-ads-detection/ItemInfo_train_processed.csv"
# reader = pd.read_csv(train_info_path, chunksize=10, encoding='utf-8')
#
# for batch in reader:
#     print(batch['description'])
#     text_preprocess(batch['description'])
#     break