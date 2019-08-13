from preprocessing.preprocessing_util import add_length_column, remove_pairs

# add_length_column(pairs_file_path='dataset/avito-duplicate-ads-detection/ItemPairs_test_processed.csv',
#                   data_file_path='dataset/fasttext_data.hdf5', id_do_data_map='dataset/id_to_desc_map.pickle')
remove_pairs(pairs_file_path='dataset/ItemPairs_train_processed.csv',
             data_file_path='dataset/fasttext_data.hdf5',
             ids_to_delete=[11, 17, 30, 55, 5, 62, 90, 91, 92, 95])
# import plot_precision_recall_curve as pprc
# import pandas as pd
# from sklearn.metrics import average_precision_score
# import matplotlib.pyplot as plt

# test_set = pd.read_csv('dataset/avito-duplicate-ads-detection/ItemPairs_test_processed.csv')
# test_predict = pd.read_csv('experiments/standard_8_experiment/result_outputs/test_predictions.csv')
# test_y = test_set['isDuplicate']
# avg_length = test_set['avg_length']
# test_y_pred = test_predict['y_pred']
# # pprc.plot_stats(test_y, test_y_pred, 0.6834, 'standard_8_pr_plot.pdf')
# # pprc.plot_loss('experiments/standard_14_experiment/result_outputs/summary.csv')
# temp = pd.concat([test_y, test_y_pred, avg_length], axis=1)
# fu = lambda x: int(x/10)
# temp['avg_length'] = temp['avg_length'].apply(fu)
# temp_1 = temp.groupby('avg_length')
# groups = temp_1.groups
# avg_precisions = []
# for item in list(groups):
#     temp_df = temp_1.get_group(item)
#     avg_precisions.append(average_precision_score(y_true=temp_df['isDuplicate'], y_score=temp_df['y_pred']))
#
#
# plt.xlabel('Sentence length / 10')
# plt.ylabel('Average Precision Score')
# #plt.title('')
# plt.bar(list(groups), avg_precisions)
# plt.savefig('standard_8_avg_aps_per_sentence.pdf')
# plt.show()
del_list = [11, 17, 30, 55, 5, 62, 90, 91, 92, 95]
