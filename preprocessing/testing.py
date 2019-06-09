import pandas as pd
import numpy as np
import string
import re
from pymystem3 import Mystem

punctuations = [x for x in string.punctuation]


def remove_punctuation(sentence):
    words = sentence.split()
    new_sentence_ = ""
    for word in words:
        new_word = ""
        for x in word:
            if not (x in punctuations):
                new_word = new_word + x
        new_sentence_ = new_sentence_ + " " + new_word

    return new_sentence_.strip()


def text_preprocess(descriptions):
    processed_desc = []
    mystem = Mystem()

    for description in descriptions:
        description = description.lower()
        description = remove_punctuation(description)
        description = mystem.lemmatize(description)
        description = "".join(description)
        description = re.sub('\n', '', description)
        processed_desc.append(description)

    return processed_desc


def select_random_image(array):
    images_list = []

    for element in array:
        images_array = list(map(int, element.split(",")))
        images_list.append(images_array[np.random.randint(0, len(images_array))])

    return images_list


train_info_path = "dataset/ItemInfo_train_processed.csv"
reader = pd.read_csv(train_info_path, chunksize=10, encoding='utf-8')

item_id = []
descriptions = []
image_id = []
i = 0
for batch in reader:
    item_id.append(batch['itemID'].tolist())
    image_id.append(select_random_image(batch['images_array']))
    descriptions.append(text_preprocess(batch['description']))

item_id = np.array(item_id)
item_id = np.reshape(item_id, item_id.shape[0] * item_id.shape[1])
image_id = np.array(image_id)
image_id = np.reshape(image_id, image_id.shape[0] * image_id.shape[1])
descriptions = np.array(descriptions)
descriptions = np.reshape(descriptions, descriptions.shape[0] * descriptions.shape[1])

columns = batch.columns
columns = list(columns)[2:]
d = {}
for col in columns:
    d[col] = []

d[columns[0]] = item_id
d[columns[1]] = descriptions
d[columns[2]] = image_id
pd.DataFrame(data=d).to_csv("dataset/ItemInfo_train_processed_1.csv")
