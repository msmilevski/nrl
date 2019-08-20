import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from plot_precision_recall_curve import plot_stats, plot_2d_points

train_data = pd.read_csv('dataset/Item_pairs_features_train.csv')
val_data = pd.read_csv('dataset/Item_pairs_features_val.csv')
test_data = pd.read_csv('dataset/Item_pairs_features_test.csv')

print("Dataset loaded.")

X = train_data[['feature_1', 'feature_2']].values
y = train_data[['y']].values
y = y.reshape(y.shape[0])

mean_feature_1 = np.mean(X[:, 0])
mean_feature_2 = np.mean(X[:, 1])
std_feature_1 = np.std(X[:, 0], ddof=1)
std_feature_2 = np.std(X[:, 1], ddof=1)

X[:, 0] = (X[:, 0] - mean_feature_1)/std_feature_1
X[:, 1] = (X[:, 1] - mean_feature_2)/std_feature_2

X_val = val_data[['feature_1', 'feature_2']].values
y_val = val_data[['y']].values
y_val = y_val.reshape(y_val.shape[0])

X_val[:, 0] = (X_val[:, 0] - mean_feature_1)/std_feature_1
X_val[:, 1] = (X_val[:, 1] - mean_feature_2)/std_feature_2

X_test = test_data[['feature_1', 'feature_2']].values
y_test = test_data[['y']].values
y_test = y_test.reshape(y_test.shape[0])

X_test[:, 0] = (X_test[:, 0] - mean_feature_1)/std_feature_1
X_test[:, 1] = (X_test[:, 1] - mean_feature_2)/std_feature_2


print("Datasets normalized.")
print("Start training logistic regression model.")
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X, y)
print("Training is finished.")

# y_pred = clf.predict(X)
# y_pred = y_pred.reshape(y_pred.shape[0])
# average_precision = average_precision_score(y_true=y, y_score=y_pred)
# print("Average precision score train set: " + str(average_precision))
# plot_stats(y, y_pred, average_precision, figure_name='baseline_train.pdf')
# plot_2d_points(X, y, fig_name='baseline_data.png')
# plot_2d_points(X, y_pred, fig_name='baseline_predictions.png')

# y_pred = clf.predict(X_val)
# y_pred = y_pred.reshape(y_pred.shape[0])
# average_precision = average_precision_score(y_true=y_val, y_score=y_pred)
# print("Average precision score val set: " + str(average_precision))
# plot_stats(y_val, y_pred, average_precision, figure_name='baseline_val.pdf')
# plot_2d_points(X_val, y_val, fig_name='baseline_val_data.png')
# plot_2d_points(X_val, y_pred, fig_name='baseline_val_predictions.png')

y_pred = clf.predict(X_test)
y_pred = y_pred.reshape(y_pred.shape[0])
#average_precision = average_precision_score(y_true=y_test, y_score=y_pred)
print("Average precision score test set: " + str(0))
plot_stats(y_test, y_pred, 0, figure_name='baseline_test.pdf')
#plot_2d_points(X_test, y_test, fig_name='baseline_test_data.png')
#plot_2d_points(X_test, y_pred, fig_name='baseline_test_predictions.png')
column = pd.DataFrame({'y_pred': y_pred})
column.to_csv('baseline_test_predictions.csv', encoding='utf-8')