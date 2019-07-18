import h5py
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
# Ignore warinings
import warnings
import ast
import pickle
import preprocessing.preprocessing_util as util

warnings.filterwarnings("ignore")


class DatasetProvider(Dataset):

    def __init__(self, pair_file_path, data_file_path, images_dir, transform=None, start_id=-1):
        '''
        Class that creates the dataset online, using the ids from the pairs dataset
        :param pair_file_path: path to the Pairs dataset
        :param data_file_path: path to the description dataset
        :param images_dir: path to the directory where the image embeddings are stored
        :param transform: PyTorch transform object that it's used for transforming our input to a tensor
        '''
        # Read data
        self.pairs = pd.read_csv(pair_file_path, encoding='utf-8')
        if start_id != -1:
            start_index = start_id*207551
            end_index = (start_id + 1)*207551
            if end_index > len(self.pairs):
                end_index = len(self.pairs) + 1
            self.pairs = self.pairs[start_index : end_index]

        data = h5py.File(data_file_path, 'r')
        self.images_dir = images_dir
        self.transform = transform
        self.item_idx = data['itemID'][()]
        self.image_ids = data['image_id'][()]
        self.descriptions = data['descriptions'][()]

    def get_image_embedding(self, image_id):
        folder_id = image_id % 100
        with h5py.File(self.images_dir + "/image_features_" + str(folder_id) + ".hdf5", 'r') as img_data:
        # with h5py.File(self.images_dir, 'r') as img_data:
            # i ovie treba da se smenat za baseline
            ids = img_data['image_id'][()]
            position_item = np.argwhere(ids == image_id)[0][0]
            return img_data['image_features'][position_item]
            # ids = img_data['img_id'][()]
            # position_item = np.argwhere(ids == image_id)[0][0]
            # return img_data['img_embed'][position_item]

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]

        y = int(pair['isDuplicate'])

        item_1_id = int(pair['itemID_1'])
        item_2_id = int(pair['itemID_2'])

        position_item_1 = np.argwhere(self.item_idx == item_1_id)[0][0]
        position_item_2 = np.argwhere(self.item_idx == item_2_id)[0][0]

        item_1_desc = self.descriptions[position_item_1]
        item_2_desc = self.descriptions[position_item_2]
        item_1_img = self.image_ids[position_item_1]
        item_2_img = self.image_ids[position_item_2]

        img_1 = self.get_image_embedding(int(item_1_img))
        img_2 = self.get_image_embedding(int(item_2_img))

        return {'desc1': item_1_desc, 'image_1': img_1, 'desc2': item_2_desc, 'image_2': img_2, 'target': y}
