import os
from glob import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
import pickle

# Ignore warinings
import warnings

warnings.filterwarnings("ignore")

import arg_extractor


class ImageDataset(Dataset):
    '''Images dataset'''

    def __init__(self, image_paths, transform=None):
        '''
        :param root_dir: Root directory from which we will open the images
        :param transform: Optional transform to be applied on a image
        '''
        self.transform = transform
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = int(img_path.split("/")[-1].split(".")[0])
        image = cv2.imread(img_path)

        if not(image is None):
            image = image[:,:,:3]
        else:
            image = np.zeros((256,256,3),dtype=np.double)
            img_id = -1

        if self.transform:
            sample = self.transform(image)
        return {'image': sample, 'image_id': img_id}


class Rescale(object):
    """Rescale the image in a sample to a given size."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_h, new_w))
        return img


class RandomCrop(object):
    """Crop randomly the image in a sample."""

    def __init__(self, output_size):
        """
        :param output_size: Desired output size. If int, square crop is made.
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return image


args, device = arg_extractor.get_args()
print(args)

dict = pickle.load(open('dataset/Image_embed_dict.pickle', 'rb'))
arr = list(dict.keys())[args.seed*5 : (args.seed+1)*5]

composed = transforms.Compose([Rescale(256),
                               RandomCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

alexnet = models.alexnet(pretrained=True)
alexnet = nn.Sequential(*list(alexnet.children())[:-1])
alexnet = alexnet
alexnet.to(device)

general_path = "/home/s1885778/nrl/dataset/Images_/Images_"

for item in arr:
    image_paths = []
    image_ids = dict[item]
    folder = int(int(item) / 10)
    for id in image_ids:
        temp = general_path + str(folder) + '/' + str(item) + "/" + str(id) + '.jpg'
        image_paths.append(temp)


    image_dataset = ImageDataset(image_paths=image_paths, transform=composed)
    print(len(image_dataset))
    dataload = DataLoader(image_dataset, batch_size=args.batch_size, num_workers=0)


    features = []
    ids = []
    for i_batch, sample_batched in enumerate(dataload):
        # Get image features
        print("Put images on device: " + str(device))
        input = sample_batched['image'].    to(device)
        print("Put them through the pretrained network...")
        batch_features = alexnet.forward(input)
        # Reshape output from the last layer of the resnet
        print("Return data on CPU")
        batch_features = batch_features.cpu()
        print(batch_features.shape)
        batch_features = batch_features.reshape(batch_features.shape[0], batch_features.shape[1])
        # Use detach to imply that I don't need gradients
        # Turn tensor into numpy array
        # Save each image feature with its corresponing img_id
        print("Add batch to list...")
        batch_features = batch_features.detach().numpy().astype(float)
        for i, id in enumerate(sample_batched['image_id']):
            if int(id) != -1:
                ids.append(int(id))
                features.append(batch_features[i, :])
            else:
                print(id)

    # Saving the data
    save_file_path = "/home/s1885778/nrl/dataset/alexnet/image_features_" + str(item) + ".hdf5"
    print("Saving file: " + save_file_path + " ...")
    data_file = h5py.File(save_file_path, 'w')
    data_file.create_dataset("image_id", data=ids)
    data_file.create_dataset("image_features", data=features)
