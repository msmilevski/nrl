import h5py
import pandas as pd
import numpy as np
import sys

pair_file_paths = ['dataset/subsampledItemPairs_test_processed.csv', 'dataset/subsampledItemPairs_train_processed.csv',
                   'dataset/subsampledItemPairs_val_processed.csv']
data_file_path = 'dataset/fasttext_data.hdf5'
data = h5py.File(data_file_path, 'r')

item_idx = data['itemID'][()]
image_ids = data['image_id'][()]
desc = data['descriptions'][()]

items = {}
img_ids = []
desciptions = []

for path in pair_file_paths:
    pairs = pd.read_csv(path, encoding='utf-8')
    for idx in range(len(pairs)):
        pair = pairs.iloc[idx]

        item_1_id = int(pair['itemID_1'])
        item_2_id = int(pair['itemID_2'])

        if item_1_id not in items:
            items[item_1_id] = 1
            position_item_1 = np.argwhere(item_idx == item_1_id)[0][0]
            img_ids.append(image_ids[position_item_1])
            desciptions.append(desc[position_item_1])

        if item_2_id not in items:
            items[item_2_id] = 1
            position_item_2 = np.argwhere(item_idx == item_2_id)[0][0]
            item_2_img = image_ids[position_item_2]
            img_ids.append(image_ids[position_item_2])
            desciptions.append(desc[position_item_2])

file = h5py.File('subsampled_fasttext_data.hdf5', 'w')
file.create_dataset('itemID', data=list(items.keys()))
file.create_dataset('descriptions', data=desciptions)
file.create_dataset('image_id', data=img_ids)
