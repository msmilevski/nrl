from data_provider import DatasetProvider
import numpy as np
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from models import BaselineModel

args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

from torchvision import transforms
import torch
from torch.utils.data import DataLoader

torch.manual_seed(seed=args.seed)  # sets pytorch's seed

if args.dataset_name == 'baseline':
    training_data = DatasetProvider(pair_file_path='dataset/ItemPairs_train_processed.csv',
                                    data_file_path='dataset/fasttext_data.hdf5',
                                    images_dir='dataset/resnet152/', isBaseline=True)
    training_data = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_data = DatasetProvider(pair_file_path='dataset/ItemPairs_val_processed.csv',
                                    data_file_path='dataset/fasttext_data.hdf5',
                                    images_dir='dataset/resnet152/', isBaseline=True)
    valid_data = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_data = DatasetProvider(pair_file_path='dataset/ItemPairs_test_processed.csv',
                                    data_file_path='dataset/fasttext_data.hdf5',
                                    images_dir='dataset/resnet152/', isBaseline=True)
    test_data = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

elif args.dataset_name == 'standard':
    training_data = DatasetProvider(pair_file_path='dataset/ItemPairs_train_processed.csv',
                                    data_file_path='dataset/fasttext_data.hdf5',
                                    images_dir='dataset/resnet152/', isBaseline=False)
    training_data = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_data = DatasetProvider(pair_file_path='dataset/ItemPairs_val_processed.csv',
                                 data_file_path='dataset/fasttext_data.hdf5',
                                 images_dir='dataset/resnet152/', isBaseline=False)
    valid_data = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_data = DatasetProvider(pair_file_path='dataset/ItemPairs_test_processed.csv',
                                data_file_path='dataset/fasttext_data.hdf5',
                                images_dir='dataset/resnet152/', isBaseline=False)
    test_data = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

# Binary classification
num_output_classes = 2

if args.model_name == 'baseline':
    model = BaselineModel(input_dim=2)


conv_experiment = ExperimentBuilder(network_model=model,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    continue_from_epoch=args.continue_from_epoch,
                                    device=device,
                                    train_data=training_data, val_data=valid_data,
                                    test_data=test_data)  # build an experiment object
# experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
