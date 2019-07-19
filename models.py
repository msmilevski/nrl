import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class VQAStandard(nn.Module):
    def __init__(self, desc_input_shape, img_input_shape, num_output_classes, use_bias, hidden_size,
                 encoder_output_size, embedding_matrix):
        super(VQAStandard, self).__init__()
        self.desc_input_shape = desc_input_shape
        self.img_input_shape = img_input_shape
        self.num_classes = num_output_classes
        self.use_bias = use_bias
        self.hidden_size = hidden_size
        self.out_features = encoder_output_size
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

        self.layer_dict['desc_fc'] = nn.Linear(in_features=4 * self.hidden_size, out_features=self.out_features, bias=self.use_bias)
        self.layer_dict['img_fc'] = nn.Linear(in_features=self.img_input_shape[1], out_features=self.out_features,
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
    def __init__(self, item_1_model, item_2_model, encoder_output_size, fc1_size, fc2_size, use_bias=True):
        super(SiameseNetwork, self).__init__()
        self.item_1_model = item_1_model
        self.item_2_model = item_2_model
        self.bias = use_bias
        self.in_features = encoder_output_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.layer_dict = nn.ModuleDict()
        self.build_model()

    def build_model(self):
        self.layer_dict['fcn1'] = nn.Linear(in_features=self.in_features, out_features=self.fc1_size, bias=self.bias)
        self.layer_dict['fcn2'] = nn.Linear(in_features=self.fc1_size, out_features=self.fc2_size, bias=self.bias)
        self.layer_dict['fcn3'] = nn.Linear(in_features=self.fc2_size, out_features=1, bias=self.bias)

    def forward_once(self, input):
        out = input
        out = self.layer_dict['fcn1'](out)
        out = nn.ReLU(inplace=True)(out)
        out = self.layer_dict['fcn2'](out)
        out = nn.ReLU(inplace=True)(out)
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
        # Compute distance between the outputs of the Siamese model
        distance = torch.abs(output_1 - output_2)
        # Weight the component-wise distance between the two feature vectors with learnable parameters
        out = self.layer_dict['fcn3'](distance)
        # Apply a sigmoid function to the output
        out = F.sigmoid(out)

        return out

    def reset_parameters(self):
        self.item_1_model.reset_parameters()
        self.item_2_model.reset_parameters()
        for item in self.layer_dict.children():
            item.reset_parameters()

