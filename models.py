import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class VQAStandard(nn.Module):
    def __init__(self, desc_input_shape, img_input_shape, num_output_classes, use_bias, hidden_size, num_lstms,
                 encoder_output_size, embedding_matrix):
        super(VQAStandard, self).__init__()
        self.desc_input_shape = desc_input_shape
        self.img_input_shape = img_input_shape
        self.num_classes = num_output_classes
        self.use_bias = use_bias
        self.hidden_size = hidden_size
        self.out_features = encoder_output_size
        self.num_lstms = num_lstms
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
        self.layer_dict['lstm'] = nn.LSTM(input_size=out_desc.shape[-1], hidden_size=self.hidden_size,
                                          num_layers=self.num_lstms)

        self.layer_dict['desc_fc'] = nn.Linear(in_features=2 * self.num_lstms * self.hidden_size,
                                               out_features=self.out_features,
                                               bias=self.use_bias)
        self.layer_dict['img_fc'] = nn.Linear(in_features=self.img_input_shape[1], out_features=self.out_features,
                                              bias=self.use_bias)

    def forward(self, input):
        desc = input[0]
        img_embed = input[1]

        batch_size = desc.shape[0]
        out_desc = self.embedding_layer(desc)
        # Transform input from (batch_size, seq_length, embedding_size) to (seq_length, batch_size, embedding_size)
        out_desc = out_desc.reshape(out_desc.shape[1], out_desc.shape[0], out_desc.shape[2]).type(torch.float)
        out, (h, c) = self.layer_dict['lstm'](out_desc)
        out_desc = torch.cat((h, c), dim=2)
        out_desc = out_desc.reshape((out_desc.shape[1], out_desc.shape[0] * out_desc.shape[2]))
        out_desc = self.layer_dict['desc_fc'](out_desc)
        out_img = self.layer_dict['img_fc'](img_embed)
        # Point-wise multiplication
        out = out_desc * out_img

        return out

    def reset_parameters(self):
        for item in self.layer_dict.children():
            item.reset_parameters()


class StackedAttentionNetwork(nn.Module):
    def __init__(self, desc_input_shape, img_input_shape, num_output_classes, use_bias, hidden_size,
                 attention_kernel_size, num_att_layers, embedding_matrix):
        super(StackedAttentionNetwork, self).__init__()
        self.desc_input_shape = desc_input_shape
        self.img_input_shape = img_input_shape
        self.num_classes = num_output_classes
        self.use_bias = use_bias
        self.num_att_layers = num_att_layers
        self.hidden_size = hidden_size
        self.attention_kernel_size = attention_kernel_size
        self.layer_dict = nn.ModuleDict()
        self.embedding_layer = self.create_embedding_layer(embedding_matrix)
        self.build_model()

    def create_embedding_layer(self, embedding_matrix):
        embedding_matrix = torch.from_numpy(embedding_matrix)
        return nn.Embedding.from_pretrained(embeddings=embedding_matrix)

    def build_model(self):
        out_desc = torch.zeros(self.desc_input_shape, dtype=torch.long)

        out_desc = self.embedding_layer(out_desc)
        out_desc = out_desc.reshape(out_desc.shape[1], out_desc.shape[0], out_desc.shape[2]).type(torch.float)
        self.layer_dict['lstm'] = nn.LSTM(input_size=out_desc.shape[-1], hidden_size=self.hidden_size)
        h, c = self.layer_dict['lstm'](out_desc)

        out_desc = h[-1]

        out_img = torch.zeros(self.img_input_shape)
        self.num_areas = out_img.shape[2] * out_img.shape[3]
        out_img = out_img.reshape(out_img.shape[0], self.num_areas, out_img.shape[1])
        self.layer_dict['fc_transform_img'] = nn.Linear(in_features=out_img.shape[2], out_features=self.hidden_size,
                                                        bias=True)
        out_img = self.layer_dict['fc_transform_img'](out_img)

        u_k = out_desc
        for i in range(self.num_att_layers):
            self.layer_dict['fc_transform_img_{}'.format(i)] = nn.Linear(in_features=out_img.shape[-1],
                                                                         out_features=self.attention_kernel_size,
                                                                         bias=False)
            temp_out_img = self.layer_dict['fc_transform_img_{}'.format(i)](out_img)
            self.layer_dict['fc_transform_query_{}'.format(i)] = nn.Linear(in_features=u_k.shape[-1],
                                                                           out_features=self.attention_kernel_size,
                                                                           bias=True)
            temp_u = self.layer_dict['fc_transform_query_{}'.format(i)](u_k)
            temp_u = temp_u.unsqueeze(1)
            h_a = temp_out_img + temp_u
            h_a = torch.tanh(h_a)
            self.layer_dict['fc_prob_{}'.format(i)] = nn.Linear(in_features=self.attention_kernel_size,
                                                                out_features=1,
                                                                bias=True)
            p_i = self.layer_dict['fc_prob_{}'.format(i)](h_a)
            p_i = p_i
            p_i = F.softmax(p_i, dim=1)

            temp_out_img = out_img.reshape(out_img.shape[0], out_img.shape[2], out_img.shape[1])
            v_lambda = torch.bmm(temp_out_img, p_i)
            v_lambda = v_lambda.squeeze()
            u_k = v_lambda + u_k

    def forward(self, input):
        out_desc = input[0]
        out_img = input[1]
        self.num_areas = out_img.shape[2] * out_img.shape[3]

        out_desc = self.embedding_layer(out_desc)
        out_desc = out_desc.reshape(out_desc.shape[1], out_desc.shape[0], out_desc.shape[2]).type(torch.float)
        h, c = self.layer_dict['lstm'](out_desc)
        out_desc = h[-1]

        out_img = out_img.reshape(out_img.shape[0], self.num_areas, out_img.shape[1])
        out_img = self.layer_dict['fc_transform_img'](out_img)

        u_k = out_desc
        for i in range(self.num_att_layers):
            temp_out_img = self.layer_dict['fc_transform_img_{}'.format(i)](out_img)
            temp_u = self.layer_dict['fc_transform_query_{}'.format(i)](u_k)
            temp_u = temp_u.unsqueeze(1)
            h_a = temp_out_img + temp_u
            h_a = torch.tanh(h_a)
            p_i = self.layer_dict['fc_prob_{}'.format(i)](h_a)
            p_i = p_i
            p_i = F.softmax(p_i, dim=1)

            temp_out_img = out_img.reshape(out_img.shape[0], out_img.shape[2], out_img.shape[1])
            v_lambda = torch.bmm(temp_out_img, p_i)
            v_lambda = v_lambda.squeeze()
            u_k = v_lambda + u_k

        return u_k

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
        # Define the fully connencted layers
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
