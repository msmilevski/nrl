import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from math import *
from preprocessing.preprocessing_util import baseline_preprocessing


# def jaccard_similarity(x, y):
#     x = baseline_preprocessing(x)
#     y = baseline_preprocessing(y)
#     intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
#     union_cardinality = len(set.union(*[set(x), set(y)]))
#     return intersection_cardinality / float(union_cardinality)
#
#
# def euclidean_distance(x, y):
#     return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))
#
#
# class LogisticRegression(nn.Module):
#     def __init__(self, input_dim):
#         super(LogisticRegression, self).__init__()
#         self.linear = nn.Linear(input_dim, 2)
#
#     def forward(self, x):
#         y_pred = F.sigmoid(self.linear(x))
#         return y_pred
#
#     def reset_parameters(self):
#         self.linear.reset_parameters()
#
#
# class BaselineModel(nn.Module):
#     def __init__(self, input_dim):
#         super(BaselineModel, self).__init__()
#         self.build_model(input_dim=input_dim)
#         self.elements = [0, 1, 2]
#
#     def build_model(self, input_dim):
#         self.model = LogisticRegression(input_dim)
#
#     def feature_engineering(self, x):
#         # return {'desc1': item_1_desc, 'image_1': img_1, 'desc2': item_2_desc, 'image_2': img_2, 'target': y}
#         desc_1_batch = x[0]
#         desc_2_batch = x[2]
#         img_1_batch = x[1]
#         img_2_batch = x[3]
#
#         batch_x = []
#
#         for i in range(len(desc_1_batch)):
#             feature_1 = jaccard_similarity(desc_1_batch[i].numpy(), desc_2_batch[i].numpy())
#             feature_2 = euclidean_distance(img_1_batch[i].numpy(), img_2_batch[i].numpy())
#             batch_x.append([feature_1, feature_2])
#
#         batch_x = torch.from_numpy(np.array(batch_x))
#
#         return batch_x
#
#     def forward(self, x):
#         x = self.feature_engineering(x)
#         return self.model(x.float())
#
#     def reset_parameters(self):
#         self.model.reset_parameters()


class VQAStandard(nn.Module):
    def __init__(self, desc_input_shape, img_input_shape, num_output_classes, use_bias, hidden_size, embedding_matrix):
        super(VQAStandard, self).__init__()
        self.desc_input_shape = desc_input_shape
        self.img_input_shape = img_input_shape
        self.num_classes = num_output_classes
        self.use_bias = use_bias
        self.hidden_size = hidden_size
        self.layer_dict = nn.ModuleDict()
        self.embedding_layer = self.create_embedding_layer(embedding_matrix)
        self.build_model()

    def create_embedding_layer(self, embedding_matrix):
        embedding_matrix = torch.from_numpy(embedding_matrix)
        return nn.Embedding.from_pretrained(embeddings=embedding_matrix)

    def build_model(self):
        x_desc = torch.zeros(self.desc_input_shape, dtype=torch.long)
        out_desc = x_desc

        # Define Layers
        out_desc = self.embedding_layer(out_desc)
        self.layer_dict['first_lstm_cell'] = nn.LSTMCell(input_size=out_desc.shape[2], hidden_size=self.hidden_size,
                                                         bias=self.use_bias)
        self.layer_dict['second_lstm_cell'] = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                                          bias=self.use_bias)

        self.layer_dict['desc_fc'] = nn.Linear(in_features=4 * self.hidden_size, out_features=1024, bias=self.use_bias)
        self.layer_dict['img_fc'] = nn.Linear(in_features=self.img_input_shape[1], out_features=1024,
                                              bias=self.use_bias)

    def forward(self, input):
        desc = input[0]
        img_embed = input[1]

        batch_size = desc.shape[0]
        out_desc = self.embedding_layer(desc)
        # Transform input from (batch_size, seq_length, embedding_size) to (seq_length, batch_size, embedding_size)
        out_desc = out_desc.reshape(out_desc.shape[1], out_desc.shape[0], out_desc.shape[2]).type(torch.float)

        # Define random initialization for both lstm cells
        h_t1 = torch.from_numpy(np.random.randn(batch_size, self.hidden_size)).type(torch.float)
        c_t1 = torch.from_numpy(np.random.randn(batch_size, self.hidden_size)).type(torch.float)
        h_t2 = torch.from_numpy(np.random.randn(batch_size, self.hidden_size)).type(torch.float)
        c_t2 = torch.from_numpy(np.random.randn(batch_size, self.hidden_size)).type(torch.float)

        for i in range(out_desc.shape[0]):
            h_t1, c_t1 = self.layer_dict['first_lstm_cell'](out_desc[i], (h_t1, c_t1))
            h_t2, c_t2 = self.layer_dict['second_lstm_cell'](h_t1, (h_t2, c_t2))

        # Not sure about the ordering here
        out_desc = torch.cat((h_t1, c_t1, h_t2, c_t2), dim=1)

        out_desc = self.layer_dict['desc_fc'](out_desc)
        out_img = self.layer_dict['img_fc'](img_embed)
        # Point-wise multiplication
        out = out_desc * out_img

        return out

    def reset_parameters(self):
        for item in self.layer_dict.children():
            item.reset_parameters()


class SiameseNetwork(nn.Module):
    def __init__(self, item_1_model, item_2_model, use_bias=True):
        super(SiameseNetwork, self).__init__()
        self.item_1_model = item_1_model
        self.item_2_model = item_2_model
        self.bias = use_bias
        self.layer_dict = nn.ModuleDict()
        self.build_model()

    def build_model(self):
        self.layer_dict['fcn1'] = nn.Linear(in_features=1024, out_features=512, bias=self.bias)
        self.layer_dict['relu1'] = nn.ReLU(inplace=True)
        self.layer_dict['fcn2'] = nn.Linear(in_features=512, out_features=2056, bias=self.bias)
        self.layer_dict['relu2'] = nn.ReLU(inplace=True)
        self.layer_dict['fcn3'] = nn.Linear(in_features=2056, out_features=2, bias=self.bias)

    def forward_once(self, input):
        out = input
        out = self.layer_dict['fcn1'](out)
        out = self.layer_dict['relu1'](out)
        out = self.layer_dict['fcn2'](out)
        out = self.layer_dict['relu2'](out)
        out = self.layer_dict['fcn3'](out)
        return out

    def forward(self, input):
        desc_1 = input[0]
        img_embed_1 = input[1]
        desc_2 = input[2]
        img_embed_2 = input[3]

        input_1 = [desc_1, img_embed_1]
        input_2 = [desc_2, img_embed_2]

        output_1 = self.item_1_model(input_1)
        output_2 = self.item_2_model(input_2)

        output_1 = self.forward_once(output_1)
        output_2 = self.forward_once(output_2)

        return output_1, output_2

    def reset_parameteres(self):
        self.item_1_model.reset_parameters()
        self.item_2_model.reset_parameters()
        for item in self.layer_dict.children():
            item.reset_parameters()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output_1, output_2, label):
        # Find the pairwise distance or euclidian distance of two output feature vectors
        euclidian_distance = F.pairwise_distance(output_1, output_2)
        # perform constrastive loss calculation with the distsance
        loss_constrastive = torch.mean((1 - label) * torch.pow(euclidian_distance, 2) + (label) * torch.pow(
            torch.clamp(self.margin - euclidian_distance, min=0.0), 2))

        return loss_constrastive

# seed = np.random.RandomState(seed=124)
# embedding_matrix = np.load('dataset/fasttext_embed_10000.npy')
# model_1 = VQAStandard(desc_input_shape=(64, 102),
#                     img_input_shape=(64, 2048),
#                     num_output_classes=2,
#                     use_bias=True,
#                     hidden_size=512,
#                     embedding_matrix=embedding_matrix)
#
# model_2 = VQAStandard(desc_input_shape=(64, 102),
#                     img_input_shape=(64, 2048),
#                     num_output_classes=2,
#                     use_bias=True,
#                     hidden_size=512,
#                     embedding_matrix=embedding_matrix)
#
# desc, img_embed = torch.from_numpy(np.random.randint(100, size=(64, 102))).type(torch.long), \
#                   torch.from_numpy(np.random.randn(64, 2048)).type(torch.float)
#
# siamese_model = SiameseNetwork(model_1, model_2, use_bias = True)
# siamese_model([desc, img_embed, desc, img_embed])
