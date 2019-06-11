import re
from pymystem3 import Mystem
import sys
import pandas as pd
from transliterate import translit
from tqdm import tqdm
import json


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


# def preprocess_corpus(data_path, processed_text_file_path, l):
#     reader = pd.read_csv(data_path, chunksize=100, encoding='utf-8')
#     reg = re.compile('[^a-z^A-Z^0-9^А-я^\s*]')
#     lemm = (l == 'True')
#
#     item_id = []
#     descriptions = []
#
#     if lemm == True:
#         mystem = Mystem()
#
#     print(lemm)
#     for batch in tqdm(reader):
#         b_descriptions = batch['description']
#         ids = batch['itemID'].tolist()
#         temp_text = ""
#         for i, desc in enumerate(b_descriptions):
#             if lemm == True:
#                 descriptions.append(preprocess_line(desc, reg, mystem))
#             else:
#                 descriptions.append(preprocess_line(desc, reg))
#             item_id.append(ids[i])
#
#     d = {}
#     d['itemID'] = item_id
#     d['descriptions'] = descriptions
#     pd.DataFrame(data=d).to_csv(processed_text_file_path)

# def create_word_frequency_document(self, path_to_json_file='../dataset/word_frequencies.json'):
#
#     data = json.load(open(self.path_to_file))
#     annotations = data['annotations']
#
#     frequency = {}
#     for annotation in annotations:
#         sentence = annotation[0]['text'].split()
#         for word in sentence:
#             # proverka za brishenje na greski so zborovi vo unicode format(latinski zborovi)
#             if any(x.isupper() for x in unidecode(word)) == False:
#                 count = frequency.get(word, 0)
#                 frequency[word] = count + 1
#
#     sorted_frequency = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
#
#     with open(path_to_json_file, 'w') as fp:
#         json.dump(sorted_frequency, fp)

# def get_n_most_frequent_words(self, word_frequency_file='../dataset/word_frequencies.json', vocabulary_size=10000):
#
#         data = json.load(open(word_frequency_file))
#         return data[0:vocabulary_size]
#
# def generate_vocabulary(self, vocabulary_file='../dataset/vist2017_vocabulary.json',
#                             word_frequency_file='../dataset/word_frequencies.json', vocabulary_size=10000):
#
#         data = self.get_n_most_frequent_words(word_frequency_file, vocabulary_size)
#
#         idx_to_words = []
#         idx_to_words.append("<NULL>")
#         idx_to_words.append("<START>")
#         idx_to_words.append("<END>")
#         idx_to_words.append("<UNK>")
#
#         for element in data:
#             idx_to_words.append(element[0])
#
#         words_to_idx = {}
#         for i in range(len(idx_to_words)):
#             words_to_idx[idx_to_words[i]] = i
#
#         vocabulary = {}
#         vocabulary["idx_to_words"] = idx_to_words
#         vocabulary["words_to_idx"] = words_to_idx
#
#         with open(vocabulary_file, 'w') as fp:
#             json.dump(vocabulary, fp)
#
# def sentences_to_index_helper(self, sentence, word_to_idx, max_length):
#         words = sentence.split()
#         result_sentence = []
#
#         for word in words:
#             if len(result_sentence) == max_length:
#                 break
#             else:
#                 if (word_to_idx.has_key(word)):
#                     result_sentence.append(word_to_idx[word])
#                 else:
#                     result_sentence.append(word_to_idx["<UNK>"])
#
#         result_sentence.insert(0, word_to_idx["<START>"])
#         result_sentence.append(word_to_idx["<END>"])
#
#         while len(result_sentence) < max_length + 2:
#             result_sentence.append(word_to_idx["<NULL>"])
#
#         return result_sentence
#
# def indecies_to_sentence(self, sentence, idx_to_word):
#
#         result_sentence = ""
#         for word in sentence:
#             if word == 0:
#                 result_sentence = result_sentence + " " + idx_to_word[word]
#
#         print(result_sentence)
#         return result_sentence

#preprocess_corpus(sys.argv[1], sys.argv[2], sys.argv[3])
