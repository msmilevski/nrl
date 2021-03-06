import numpy as np
import h5py
import pandas as pd
import sys


def get_image_embedding(images_dir, image_id):
    folder_id = image_id % 100
    with h5py.File(images_dir + "/image_features_" + str(folder_id) + ".hdf5", 'r') as img_data:
        ids = img_data['image_id'][()]
        position_item = np.argwhere(ids == image_id)
        print(image_id)
        print(folder_id)
        print(position_item)
        position_item = position_item[0][0]
        return img_data['image_features'][position_item]


pair_file_path = sys.argv[1]
desc_file_path = sys.argv[2]
img_embed_dir = sys.argv[3]
type_of_set = sys.argv[4]
pairs = pd.read_csv(pair_file_path, encoding='utf-8')
data = h5py.File(desc_file_path, 'r')

item_idx = data['itemID'][()]
image_ids = data['image_id'][()]
img_embed = {}

for idx in range(len(pairs)):
    pair = pairs.iloc[idx]

    item_1_id = int(pair['itemID_1'])
    item_2_id = int(pair['itemID_2'])

    position_item_1 = np.argwhere(item_idx == item_1_id)[0][0]
    position_item_2 = np.argwhere(item_idx == item_2_id)[0][0]

    item_1_img = image_ids[position_item_1]
    item_2_img = image_ids[position_item_2]

    if not(item_1_img in img_embed):
        img_embed[item_1_img] = get_image_embedding(img_embed_dir, item_1_img)

    if not(item_2_img in img_embed):
        img_embed[item_2_img] = get_image_embedding(img_embed_dir, item_2_img)


file = h5py.File('subsampled_' + type_of_set + '_img_embed.hdf5', 'w')
file.create_dataset('img_id', data=list(img_embed.keys()))
file.create_dataset('img_embed', data=list(img_embed.values()))
