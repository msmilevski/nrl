import h5py
import pandas as pd
import sys
import preprocessing_util as preprocessing

# Take arguments for script
data_file_path = sys.argv[1]
indecies_file_path = sys.argv[2]
num_subsamples = int(sys.argv[3])
do_split = (sys.argv[4] == 'True')
# Read file
print("Read indecies file:" + indecies_file_path)
ids = h5py.File(indecies_file_path, 'r')['itemID'][:].tolist()
dataframe = pd.read_csv(data_file_path, encoding='utf-8')

# Remove pairs that are not present in the description dataset
dataframe = preprocessing.remove_pairs(dataframe, ids)

if num_subsamples != -1:
    # Subsample the dataset
    print("Subsampling data to " + str(num_subsamples) + " samples.")
    dataframe = preprocessing.subsample_data(dataframe, num_subsamples)

if do_split:
    print("Split the dataset to training and validation...")
    # Make train valid split
    train_dataset, val_dataset = preprocessing.split_to_train_val(dataframe)
    train_dataset.to_csv("/dataset/avito-duplicate-ads-detection/ItemPairs_train_processed.csv", encoding='utf-8')
    val_dataset.to_csv("/dataset/avito-duplicate-ads-detection/ItemPairs_val_processed.csv", encoding='utf-8')
else:
    print("Heyooo")
    dataframe.to_csv("/dataset/avito-duplicate-ads-detection/ItemPairs_test_processed.csv", encoding='utf-8')
