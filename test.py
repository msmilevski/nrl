from torch.utils.data import DataLoader
from data_provider import DatasetProvider
from torchvision import transforms, utils
import sys

composed = transforms.ToTensor()
dataset = DatasetProvider(pair_file_path='dataset/subsampleItemPairs_train_processed.csv',
                          data_file_path='dataset/fasttext_data_baseline.hdf5',
                          images_dir='dataset/resnet152/')
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
for i_batch, sample_batched in enumerate(dataloader):
    print(sample_batched)
    break