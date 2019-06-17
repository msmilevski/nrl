import re
from pymystem3 import Mystem
import sys
import pandas as pd
from transliterate import translit
from tqdm import tqdm
import json
import operator
import h5py

def preprocess_line(line, reg, mystem=None):
    line = line.lower()
    line = re.sub('\d+', ' 0 ', line)
    line = re.sub('\.', '. ', line)
    line = reg.sub("", line)
    line = re.sub('\s\s+', ' ', line)
    line = re.sub("\n", '', line)

    line = line.strip()
    line = translit(line, 'ru')

    if (mystem != None):
        line = mystem.lemmatize(line)
        line = "".join(line)

    return line


def preprocess_corpus(data_path, processed_text_file_path, l):
    reader = pd.read_csv(data_path, chunksize=100, encoding='utf-8')
    reg = re.compile('[^a-z^A-Z^0-9^А-я^\s*]')
    lemm = (l == 'True')

    item_id = []
    descriptions = []

    if lemm == True:
        mystem = Mystem()

    print(lemm)
    for batch in tqdm(reader):
        b_descriptions = batch['description']
        ids = batch['itemID'].tolist()
        temp_text = ""
        for i, desc in enumerate(b_descriptions):
            if lemm == True:
                descriptions.append(preprocess_line(desc, reg, mystem))
            else:
                descriptions.append(preprocess_line(desc, reg))
            item_id.append(ids[i])

    d = {}
    d['itemID'] = item_id
    d['descriptions'] = descriptions
    pd.DataFrame(data=d).to_csv(processed_text_file_path)


def create_word_frequency_document(path_to_file, path_to_json_file='../dataset/word_frequencies.json'):
    reader = pd.read_csv(path_to_file, chunksize=100, encoding='utf-8')

    frequency = {}
    for batch in tqdm(reader):
        descriptions = batch['descriptions']

        for desc in descriptions:
            for word in desc.split():
                if word in frequency:
                    frequency[word] = frequency[word] + 1
                else:
                    frequency[word] = 1

    sorted_frequency = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)

    with open(path_to_json_file, 'w') as fp:
        json.dump(sorted_frequency, fp)


def get_n_most_frequent_words(word_frequency_file='../dataset/word_frequencies.json', vocabulary_size=10000):
    data = json.load(open(word_frequency_file))
    return data[0:vocabulary_size]


def generate_vocabulary(vocabulary_file='../dataset/vist2017_vocabulary.json',
                        word_frequency_file='../dataset/word_frequencies.json', vocabulary_size=10000):
    data = get_n_most_frequent_words(word_frequency_file, vocabulary_size)

    idx_to_words = []
    idx_to_words.append("<NULL>")
    idx_to_words.append("<START>")
    idx_to_words.append("<END>")
    idx_to_words.append("<UNK>")

    for element in data:
        idx_to_words.append(element[0])

    words_to_idx = {}
    for i in range(len(idx_to_words)):
        words_to_idx[idx_to_words[i]] = i

    vocabulary = {}
    vocabulary["idx_to_words"] = idx_to_words
    vocabulary["words_to_idx"] = words_to_idx

    with open(vocabulary_file, 'w') as fp:
        json.dump(vocabulary, fp)


def sent_to_idx_helper(sentence, word_to_idx, max_length):
    words = sentence.split()
    result_sentence = []
    for word in words:
        if len(result_sentence) == max_length:
            break
        else:
            if (word_to_idx.has_key(word)):
                result_sentence.append(word_to_idx[word])
            else:
                result_sentence.append(word_to_idx["<UNK>"])

    result_sentence.insert(0, word_to_idx["<START>"])
    result_sentence.append(word_to_idx["<END>"])

    while len(result_sentence) < max_length + 2:
        result_sentence.append(word_to_idx["<NULL>"])

    return result_sentence


def sentences_to_indecies(dataset_file_path, processed_text_file_path, vocab_file, max_length):
    reader = pd.read_csv(dataset_file_path, chunksize=100, encoding='utf-8')
    data = json.load(open(vocab_file))
    word_to_idx = data['word_to_idx']
    item_id = []
    descriptions = []

    for batch in tqdm(reader):
        batch_descriptions = batch['descriptions']
        ids = batch['itemID'].tolist()

        for i, desc in enumerate(batch_descriptions):
            item_id.append(ids[i])
            descriptions.append(desc, word_to_idx, max_length)

    print("Saving file: " + processed_text_file_path + " ...")
    data_file = h5py.File(processed_text_file_path, 'w')
    data_file.create_dataset("itemID", data=item_id)
    data_file.create_dataset("descriptions", data=descriptions)


# def indecies_to_sentence(self, sentence, idx_to_word):
#
#         result_sentence = ""
#         for word in sentence:
#             if word == 0:
#                 result_sentence = result_sentence + " " + idx_to_word[word]
#
#         print(result_sentence)
#         return result_sentence

# preprocess_corpus(sys.argv[1], sys.argv[2], sys.argv[3])
# create_word_frequency_document(sys.argv[1], sys.argv[2])
# generate_vocabulary(sys.argv[1], sys.argv[2], int(sys.argv[3]))
sentences_to_indecies(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))