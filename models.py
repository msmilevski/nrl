import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
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


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

    def reset_parameters(self):
        self.linear.reset_parameters()


class BaselineModel(nn.Module):
    def __init__(self, input_dim):
        super(BaselineModel, self).__init__()
        self.build_model(input_dim=input_dim)
        self.elements = [0, 1, 2]

    def build_model(self, input_dim):
        self.model = LogisticRegression(input_dim)

    def feature_engineering(self, x):
        # return {'desc1': item_1_desc, 'image_1': img_1, 'desc2': item_2_desc, 'image_2': img_2, 'target': y}
        desc_1_batch = x[0]
        desc_2_batch = x[2]
        img_1_batch = x[1]
        img_2_batch = x[3]

        batch_x = []

        for i in range(len(desc_1_batch)):
            feature_1 = jaccard_similarity(desc_1_batch[i].numpy(), desc_2_batch[i].numpy())
            feature_2 = euclidean_distance(img_1_batch[i].numpy(), img_2_batch[i].numpy())
            batch_x.append([feature_1, feature_2])

        batch_x = torch.from_numpy(np.array(batch_x))

        return batch_x

    def forward(self, x):
        x = self.feature_engineering(x)
        return self.model(x.float())

    def reset_parameters(self):
        self.model.reset_parameters()
