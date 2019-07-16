import pandas as pd
import preprocessing_util as preprocessing

files = ['dataset/avito-duplicate-ads-detection/ItemPairs_train_processed.csv',
         'dataset/avito-duplicate-ads-detection/ItemPairs_val_processed.csv',
         'dataset/avito-duplicate-ads-detection/ItemPairs_test_processed.csv']
num_instances = [10000, 1000, 1000]

for i, file_path in enumerate(files):
    df = pd.read_csv(file_path)
    df = preprocessing.subsample_data(df, num_instances=num_instances[i])
    df.to_csv('dataset/subsampled'+file_path.split('/')[-1])