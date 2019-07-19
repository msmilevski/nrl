from data_provider import DatasetProvider
import numpy as np
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from models import SiameseNetwork, VQAStandard

args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

from torchvision import transforms
import torch
from torch.utils.data import DataLoader

torch.manual_seed(seed=args.seed)  # sets pytorch's seed

print("Loading datasets ...")
print(args.dataset_name)
# Initialize random seed
seed = np.random.RandomState(seed=args.seed)

if args.dataset_name == 'standard':
    training_data = DatasetProvider(pair_file_path='dataset/ItemPairs_train_processed.csv',
                                    data_file_path='/disk/scratch/s1885778/dataset/fasttext_data.hdf5',
                                    images_dir='/disk/scratch/s1885778/dataset/resnet152_1')
    training_data = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print('Training set loaded.')
    valid_data = DatasetProvider(pair_file_path='dataset/ItemPairs_val_processed.csv',
                                 data_file_path='/disk/scratch/s1885778/dataset/fasttext_data.hdf5',
                                 images_dir='/disk/scratch/s1885778/dataset/resnet152_1')
    valid_data = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print('Validation set loaded.')
    test_data = DatasetProvider(pair_file_path='dataset/ItemPairs_test_processed.csv',
                                data_file_path='/disk/scratch/s1885778/dataset/fasttext_data.hdf5',
                                images_dir='/disk/scratch/s1885778/dataset/resnet152_1')
    test_data = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print('Test set loaded.')

# Binary classification
num_output_classes = 2
embedding_matrix = np.load('dataset/fasttext_embed_10000.npy')
if args.model_name == 'standard':
    model_1 = VQAStandard(desc_input_shape=(args.batch_size, 102),
                        img_input_shape=(args.batch_size, 2048),
                        num_output_classes=num_output_classes,
                        use_bias=True,
                        hidden_size=args.lstm_hidden_dim,
                        encoder_output_size=args.encoder_output_size,
                        embedding_matrix=embedding_matrix)

    model_2 = VQAStandard(desc_input_shape=(args.batch_size, 102),
                        img_input_shape=(args.batch_size, 2048),
                        num_output_classes=num_output_classes,
                        use_bias=True,
                        hidden_size=args.lstm_hidden_dim,
                        encoder_output_size=args.encoder_output_size,
                        embedding_matrix=embedding_matrix)


siamese_model = SiameseNetwork(model_1, model_2, encoder_output_size=args.encoder_output_size, use_bias = True)

experiment = ExperimentBuilder(network_model=siamese_model,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    continue_from_epoch=args.continue_from_epoch,
                                    device=device,
                                    train_data=training_data,
                                    val_data=valid_data,
                                    test_data=test_data)  # build an experiment object
experiment_metrics, test_metrics = experiment.run_experiment()  # run experiment and return experiment metrics
