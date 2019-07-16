import h5py
import pandas as pd
import sys
import preprocessing_util as preprocessing

# Take arguments for script
data_file_path = sys.argv[1]
indecies_file_path = sys.argv[2]

# Read file
print("Read indecies file:" + indecies_file_path)
ids = h5py.File(indecies_file_path, 'r')['itemID'][:].tolist()
dataframe = pd.read_csv(data_file_path, encoding='utf-8')

# Remove pairs that are not present in the description dataset
dataframe = preprocessing.remove_pairs(dataframe, ids)

print("Split the dataset to training, validation and test sets.")
# Make train valid test split
train_dataset, val_dataset, test_dataset = preprocessing.split_to_train_val_test(dataframe)
train_dataset.to_csv("dataset/ItemPairs_train_processed.csv", encoding='utf-8')
val_dataset.to_csv("dataset/ItemPairs_val_processed.csv", encoding='utf-8')
test_dataset.to_csv("dataset/ItemPairs_test_processed.csv", encoding='utf-8')
