from data_provider import DatasetProvider
import numpy as np
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from models import SiameseNetwork, VQAStandard, StackedAttentionNetwork

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
    image_dir = '/disk/scratch/s1885778/dataset/resnet152_1'
elif args.dataset_name == 'san':
    image_dir = '/disk/scratch/s1885778/dataset/alexnet'

training_data = DatasetProvider(pair_file_path='dataset/ItemPairs_train_processed.csv',
                                data_file_path='/disk/scratch/s1885778/dataset/fasttext_data.hdf5',
                                images_dir=image_dir)
training_data_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                  collate_fn=training_data.collater)
print('Training set loaded.')
valid_data = DatasetProvider(pair_file_path='dataset/ItemPairs_val_processed.csv',
                             data_file_path='/disk/scratch/s1885778/dataset/fasttext_data.hdf5',
                             images_dir=image_dir)
valid_data_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=2,
                               collate_fn=valid_data.collater)
print('Validation set loaded.')
test_data = DatasetProvider(pair_file_path='dataset/ItemPairs_test_processed.csv',
                            data_file_path='/disk/scratch/s1885778/dataset/fasttext_data.hdf5',
                            images_dir=image_dir)
test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=2,
                              collate_fn=test_data.collater)
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
                          num_recurrent_layers=args.num_layers,
                          encoder_output_size=args.encoder_output_size,
                          embedding_matrix=embedding_matrix)

    model_2 = VQAStandard(desc_input_shape=(args.batch_size, 102),
                          img_input_shape=(args.batch_size, 2048),
                          num_output_classes=num_output_classes,
                          use_bias=True,
                          hidden_size=args.lstm_hidden_dim,
                          num_recurrent_layers=args.num_layers,
                          encoder_output_size=args.encoder_output_size,
                          embedding_matrix=embedding_matrix)
elif args.model_name == 'san':
    model_1 = StackedAttentionNetwork(desc_input_shape=(args.batch_size, 102),
                                      img_input_shape=(args.batch_size, 512, 14, 14),
                                      num_output_classes=num_output_classes,
                                      hidden_size=100,
                                      attention_kernel_size=50,
                                      use_bias=True,
                                      num_att_layers=2,
                                      embedding_matrix=embedding_matrix)

    model_2 = StackedAttentionNetwork(desc_input_shape=(args.batch_size, 102),
                                      img_input_shape=(args.batch_size, 512, 14, 14),
                                      num_output_classes=2,
                                      hidden_size=args.lstm_hidden_dim,
                                      attention_kernel_size=args.encoder_output_size,
                                      use_bias=True,
                                      num_att_layers=2,
                                      embedding_matrix=embedding_matrix)

siamese_model = SiameseNetwork(item_1_model=model_1, item_2_model=model_2, encoder_output_size=args.encoder_output_size,
                               fc1_size=args.fc1_size, fc2_size=args.fc2_size, use_bias=True)

experiment = ExperimentBuilder(network_model=siamese_model,
                               experiment_name=args.experiment_name,
                               num_epochs=args.num_epochs,
                               learning_rate=args.lr,
                               weight_decay_coefficient=args.weight_decay_coefficient,
                               continue_from_epoch=args.continue_from_epoch,
                               device=device,
                               train_data=training_data_loader,
                               val_data=valid_data_loader,
                               test_data=test_data_loader)  # build an experiment object
experiment_metrics, test_metrics = experiment.run_experiment()  # run experiment and return experiment metrics
