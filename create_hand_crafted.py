import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data_provider import DatasetProvider
from math import *
from preprocessing.preprocessing_util import baseline_preprocessing

def jaccard_similarity(x, y):
    x = baseline_preprocessing(x)
    y = baseline_preprocessing(y)
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)


def euclidean_distance(x, y):
    return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

dataset = DatasetProvider(pair_file_path='dataset/ItemPairs_train_processed.csv',
                          data_file_path='dataset/fasttext_data.hdf5',
                          images_dir='dataset/resnet152')

print(len(dataset))
dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=2)

feature_1 = []
feature_2 = []
y = []
for i_batch, sample_batched in enumerate(dataloader):
    desc_1_batch = sample_batched['desc1']
    desc_2_batch = sample_batched['desc2']
    img_1_batch = sample_batched['image_1']
    img_2_batch = sample_batched['image_2']
    y_batch = sample_batched['target']

    for i in range(len(desc_1_batch)):
        feature_1.append(jaccard_similarity(desc_1_batch[i].numpy(), desc_2_batch[i].numpy()))
        feature_2.append(euclidean_distance(img_1_batch[i].numpy(), img_2_batch[i].numpy()))
        y.append(y_batch[i])

data = {}
data['feature_1'] = np.array(feature_1)
data['feature_2'] = np.array(feature_2)
data['y'] = np.array(y)

df = pd.DataFrame(data=data)
df.to_csv('dataset/Item_pairs_features_train.csv')