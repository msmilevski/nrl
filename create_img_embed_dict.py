import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import pickle

pairs_train = pd.read_csv('dataset/ItemPairs_train_processed.csv')
pairs_test = pd.read_csv('dataset/ItemPairs_test_processed.csv')
pairs_val = pd.read_csv('dataset/ItemPairs_val_processed.csv')
data = h5py.File('dataset/fasttext_data.hdf5', 'r')
itemID = data['itemID'][()]
image_ids = data['image_id'][()]

set1 = set(pairs_train.itemID_1)
set2 = set(pairs_train.itemID_2)
set3 = set(pairs_test.itemID_1)
set4 = set(pairs_test.itemID_2)
set5 = set(pairs_val.itemID_1)
set6 = set(pairs_val.itemID_2)

v = set1.union(set2, set3, set4, set5, set6)

valuable_image_ids = []
for k in tqdm(v):
    temp_idx = np.argwhere(itemID == k)[0][0]
    valuable_image_ids.append(image_ids[temp_idx])

dict = {}
valuable_image_ids = np.unique(valuable_image_ids)
for item in valuable_image_ids:
    folder = item % 100
    if folder in dict:
        temp = dict[folder]
        temp.append(item)
        dict[folder] = temp
    else:
        dict[folder] = [item]

with open('dataset/Image_embed_dict.pickle', 'wb') as f:
    pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)