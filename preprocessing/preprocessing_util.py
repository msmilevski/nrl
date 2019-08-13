import re
from pymystem3 import Mystem
import pandas as pd
from transliterate import translit
from tqdm import tqdm
import json
import operator
import h5py
import numpy as np
import io
import pickle


def initial_preprocess(info_path):
    print("Start of the preprocessing process for the items in: " + info_path)
    print("Reading file ...")
    reader = pd.read_csv(info_path, encoding='utf-8')
    print("Dropping columns: categoryID, title, attrsJSON, price , locationID, metroID, lat, lon")
    reader.drop(['categoryID', 'title', 'attrsJSON', 'price', 'locationID', 'metroID', 'lat', 'lon'], axis=1,
                inplace=True)

    rows_with_nan_img_array = reader.index[reader['images_array'].isna() == True].tolist()
    print("Ads without images: " + str(len(rows_with_nan_img_array)))

    rows_with_nan_desc = reader.index[reader['description'].isna() == True].tolist()
    print("Ads without description: " + str(len(rows_with_nan_desc)))

    rows_with_nan = list(set(rows_with_nan_img_array) | set(rows_with_nan_desc))
    print("Total ads to be deleted: " + str(len(rows_with_nan)))

    reader.drop(index=rows_with_nan, inplace=True)

    descriptions = reader['description']
    image_arrays = reader['images_array']
    ids = reader['itemID']

    return ids, descriptions, image_arrays


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

    line = line.strip()
    return line


def preprocess_corpus(id_data, text_data, lemmatization):
    print("Preprocessing the text corpus ...")

    reg = re.compile('[^a-z^A-Z^0-9^А-я^\s*]')

    descriptions = []
    item_ids = []
    rows_with_nan_desc = []

    id_data = id_data.tolist()
    if lemmatization == True:
        mystem = Mystem()

    print("Lemmatization: " + str(lemmatization))
    for i, descrption in tqdm(enumerate(text_data)):
        if lemmatization == True:
            temp_line = preprocess_line(descrption, reg, mystem)
        else:
            temp_line = preprocess_line(descrption, reg)

        if len(temp_line) > 0:
            descriptions.append(temp_line)
            item_ids.append(int(id_data[i]))
        else:
            rows_with_nan_desc.append(i)

    d = {}
    d['itemID'] = item_ids
    d['descriptions'] = descriptions

    d = pd.DataFrame(data=d)
    print("Number of description that are empty after preprocessing: " + str(len(rows_with_nan_desc)))

    return d, rows_with_nan_desc


def create_word_frequency_document(descriptions, path_to_json_file='../dataset/word_frequencies.json'):
    print("Computing word frequency from the text corpus ...")
    frequency = {}
    for desc in tqdm(descriptions):
        for word in desc.split():
            if word in frequency:
                frequency[word] = frequency[word] + 1
            else:
                frequency[word] = 1

    sorted_frequency = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)

    if path_to_json_file != None:
        with open(path_to_json_file, 'w') as fp:
            json.dump(sorted_frequency, fp)

    return sorted_frequency


def get_n_most_frequent_words(word_frequency_file='../dataset/word_frequencies.json', vocabulary_size=10000):
    if type(word_frequency_file) == str:
        data = json.load(open(word_frequency_file))
    else:
        data = word_frequency_file

    return data[0:vocabulary_size]


def generate_vocabulary(vocabulary_file='../dataset/vist2017_vocabulary.json',
                        word_frequency_file='../dataset/word_frequencies.json', vocabulary_size=10000):
    print("Generating the vocabulary ...")
    print("Vocabulary size: " + str(vocabulary_size))

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

    if vocabulary_file != None:
        with open(vocabulary_file, 'w') as fp:
            json.dump(vocabulary, fp)

    return vocabulary


def sent_to_idx_helper(sentence, word_to_idx, max_length):
    words = sentence.split()
    result_sentence = []
    for word in words:
        if len(result_sentence) == max_length:
            break
        else:
            if (word in word_to_idx):
                result_sentence.append(word_to_idx[word])
            else:
                result_sentence.append(word_to_idx["<UNK>"])

    result_sentence.insert(0, word_to_idx["<START>"])
    result_sentence.append(word_to_idx["<END>"])

    while len(result_sentence) < max_length + 2:
        result_sentence.append(word_to_idx["<NULL>"])

    return result_sentence


def sentences_to_indecies(id_data, text_data, image_data, processed_text_file_path, vocab_file, max_length):
    if type(vocab_file) == str:
        temp = json.load(open(vocab_file))
        word_to_idx = temp['words_to_idx']
    else:
        word_to_idx = vocab_file['words_to_idx']

    descriptions = []

    for desc in tqdm(text_data):
        descriptions.append(sent_to_idx_helper(desc, word_to_idx, max_length))

    print("Saving file: " + processed_text_file_path + " ...")
    data_file = h5py.File(processed_text_file_path, 'w')
    data_file.create_dataset("itemID", data=id_data)
    data_file.create_dataset("descriptions", data=descriptions)
    data_file.create_dataset('image_id', data=image_data)


def indecies_to_sentence(sentence, idx_to_word):
    result_sentence = ""
    for word in sentence:
        if word == 0:
            result_sentence = result_sentence + " " + idx_to_word[word]

    return result_sentence.strip()


def select_random_image(ids_array, array, seed):
    images_list = []
    np.random.seed(seed=seed)
    for element in tqdm(array):
        images_array = list(map(int, element.split(",")))
        images_list.append(images_array[np.random.randint(0, len(images_array))])

    d = {}
    d['itemID'] = ids_array
    d['image_id'] = images_list

    return pd.DataFrame(data=d)


def remove_pairs(dataframe, ids):
    print("Get all pair indecies...")
    dataframe_ids = (set(dataframe.itemID_1) | set(dataframe.itemID_2))
    ids = set(ids)
    difference = list(dataframe_ids - ids)

    rows_id1 = dataframe.index[dataframe.itemID_1.isin(difference)].tolist()
    rows_id2 = dataframe.index[dataframe.itemID_2.isin(difference)].tolist()

    rows_to_delete = list(set(rows_id1) | set(rows_id2))

    print("There are " + str(len(rows_to_delete)) + " rows in the pairs dataset that need to be deleted.")
    print("Deleting rows ...")
    dataframe.drop(index=rows_to_delete, inplace=True)

    return dataframe


def split_to_train_val_test(dataframe):
    # Get indecies for each row
    indecies = dataframe.index.values
    # Shuffle the array
    np.random.shuffle(indecies)
    length = indecies.shape[0]

    # Take the first 70% of the shuffled array
    train_indecies = indecies[0:int(length * 0.8)]
    # Take the last 30% of the shuffled array
    val_indecies = indecies[int(length * 0.8):int(length * 0.9)]
    test_indecies = indecies[int(length * 0.9):length + 1]

    # Return parts of the original dataframe
    return dataframe.loc[train_indecies], dataframe.loc[val_indecies], dataframe.loc[test_indecies]


def subsample_data(dataframe, num_instances):
    # Split the indecies from the dataframe to similar/disimilar
    df_pos = dataframe[dataframe.isDuplicate == 1].index.values
    df_neg = dataframe[dataframe.isDuplicate == 0].index.values

    r = len(df_pos) * 1.0 / len(df_neg)
    # We assume that the negative class is more prevalent
    # in the training dataset than the positive class
    choose_neg = int(num_instances * r)
    choose_pos = num_instances - choose_neg

    # Randomly choose indecies from the postive and negative set
    positive_id = np.random.choice(df_pos, choose_pos)
    negative_id = np.random.choice(df_neg, choose_neg)

    # Combine the randomly chosen indecies/rows
    subset_index = np.concatenate((positive_id, negative_id))

    result = dataframe.loc[subset_index]

    return result


def remove_element(array, element):
    return array[array != element]


def baseline_preprocessing(source_description):
    elements = [0, 1, 2]
    array = np.array(source_description)
    for elem in elements:
        array = remove_element(array, elem)

    return array


def load_vectors(fname, word_to_idx):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    vocab_size = len(word_to_idx.keys())

    embed_matrix = np.random.randn(vocab_size, d)

    found_words = 0

    print("Filling the embedding matrix has started.")
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        embedding = list(map(float, tokens[1:]))
        if word in word_to_idx:
            embed_matrix[word_to_idx[word]] = embedding
            found_words += 1

        print(found_words)
        if found_words == (vocab_size - 4):
            print("Filling up the embedding matrix has finished.")
            break

    return embed_matrix


def save_embeddings(vocab_file_path='dataset/fasttext_vocab.json',
                    pretrained_word_embed_file='dataset/cc.ru.300.vec',
                    output_file='dataset/fasttext_embed_10000.npy'):
    vocab = json.load(open(vocab_file_path))
    word_to_idx = vocab['words_to_idx']
    print("Vocab file is open.")
    embedding_matrix = load_vectors(fname=pretrained_word_embed_file, word_to_idx=word_to_idx)
    print("Embedding matrix is loaded/")
    np.save(output_file, embedding_matrix)


def add_length_column(pairs_file_path, data_file_path, id_do_data_map):
    pairs = pd.read_csv(pairs_file_path, encoding='utf-8')
    data = h5py.File(data_file_path, 'r')
    descriptions = data['descriptions'][()]
    id_to_data = pickle.load(open(id_do_data_map, 'rb'))
    avg_pair_length = []
    for idx in tqdm(range(len(pairs))):
        pair = pairs.iloc[idx]
        item_1_id = int(pair['itemID_1'])
        item_2_id = int(pair['itemID_2'])
        position_item_1 = id_to_data[item_1_id]
        position_item_2 = id_to_data[item_2_id]
        item_1_desc = descriptions[position_item_1]
        item_2_desc = descriptions[position_item_2]

        desc_1_length = (1 - (item_1_desc == 0)).sum()
        desc_2_length = (1 - (item_2_desc == 0)).sum()

        avg_pair_length.append((desc_1_length + desc_2_length) / 2)

    column = pd.DataFrame({'avg_length': avg_pair_length})
    pairs = pd.concat([pairs, column], axis=1)
    pairs.to_csv(pairs_file_path, encoding='utf-8')


def remove_pairs(pairs_file_path, data_file_path, ids_to_delete):
    pairs = pd.read_csv(pairs_file_path, encoding='utf-8')
    data = h5py.File(data_file_path, 'r')
    image_ids = data['image_id'][()]
    item_idx = data['itemID'][()]
    delete_list = []
    for idx in tqdm(range(len(pairs))):
        pair = pairs.iloc[idx]
        item_1_id = int(pair['itemID_1'])
        item_2_id = int(pair['itemID_2'])
        position_item_1 = np.argwhere(item_idx == item_1_id)[0][0]
        position_item_2 = np.argwhere(item_idx == item_2_id)[0][0]
        item_1_img = int(image_ids[position_item_1]) % 100
        item_2_img = int(image_ids[position_item_2]) % 100

        if (item_1_img in ids_to_delete) or (item_2_img in ids_to_delete):
            delete_list.append(idx)

    print(len(delete_list))
    pairs.drop(index=delete_list, inplace=True)
    pairs.to_csv("dataset/ItemPairs_test_processed_1.csv", encoding='utf-8')
    return