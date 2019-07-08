import numpy as np
import h5py
import pandas as pd
import sys


def is_image_embedding(images_dir, image_id):
    folder_id = image_id % 100
    with h5py.File(images_dir + "/image_features_" + str(folder_id) + ".hdf5", 'r') as img_data:
        ids = img_data['image_id'][()]
        position_item = np.argwhere(ids == image_id)
        if position_item == []:
            return False
        return True


pair_file_path = sys.argv[1]
desc_file_path = sys.argv[2]
img_embed_dir = sys.argv[3]
type_of_set = sys.argv[4]
pairs = pd.read_csv(pair_file_path, encoding='utf-8')
data = h5py.File(desc_file_path, 'r')

item_idx = data['itemID'][()]
image_ids = data['image_id'][()]
del_rows = []

for idx in range(len(pairs)):
    pair = pairs.iloc[idx]

    item_1_id = int(pair['itemID_1'])
    item_2_id = int(pair['itemID_2'])

    position_item_1 = np.argwhere(item_idx == item_1_id)[0][0]
    position_item_2 = np.argwhere(item_idx == item_2_id)[0][0]

    item_1_img = image_ids[position_item_1]
    item_2_img = image_ids[position_item_2]

    is_img_1 = is_image_embedding(img_embed_dir, item_1_img)
    is_img_2 = is_image_embedding(img_embed_dir, item_2_img)

    if not((is_img_1 and is_img_2)):
        del_rows.append(idx)

pairs.drop(index=del_rows, inplace=True)
pairs.to_csv(pair_file_path, encoding='utf-8')
