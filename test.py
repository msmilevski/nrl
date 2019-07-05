from torch.utils.data import DataLoader
from data_provider import DatasetProvider
from torchvision import transforms, utils
import torch
import torch.nn as nn
import sys
from models import BaselineModel
from tqdm import tqdm

composed = transforms.ToTensor()
dataset = DatasetProvider(pair_file_path='dataset/ItemPairs_train_processed.csv',
                          data_file_path='dataset/fasttext_data.hdf5',
                          images_dir='dataset/resnet152/', isBaseline=True)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

model = BaselineModel(input_dim=2)
criterion = nn.CrossEntropyLoss()
model.reset_parameters()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in tqdm(range(20)):
    model.train()
    optimizer.zero_grad()
    for i_batch, sample_batched in enumerate(dataloader):
        desc_1_batch = sample_batched['desc1']
        desc_2_batch = sample_batched['desc2']
        img_1_batch = sample_batched['image_1']
        img_2_batch = sample_batched['image_2']
        y_batch = sample_batched['target']
        x_data = [desc_1_batch, img_1_batch, desc_2_batch, img_2_batch]
        y_pred = model(x_data)
        loss = criterion(y_pred, y_batch)

        loss.backward()
        optimizer.step()

