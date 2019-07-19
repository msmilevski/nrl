import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from plot_precision_recall_curve import plot_stats

train_data = pd.read_csv('dataset/Item_pairs_features_train.csv')
val_data = pd.read_csv('dataset/Item_pairs_features_val.csv')
test_data = pd.read_csv('dataset/Item_pairs_features_test.csv')

print("Dataset loaded.")

X = train_data[['feature_1', 'feature_2']].values
y = train_data[['y']].values

mean_feature_1 = np.mean(X[:, 0])
mean_feature_2 = np.mean(X[:, 1])
std_feature_1 = np.std(X[:, 0], ddof=1)
std_feature_2 = np.std(X[:, 1], ddof=1)

X[:, 0] = (X[:, 0] - mean_feature_1)/std_feature_1
X[:, 1] = (X[:, 1] - mean_feature_2)/std_feature_2

X_val = val_data[['feature_1', 'feature_2']].values
y_val = val_data[['y']].values

X_val[:, 0] = (X_val[:, 0] - mean_feature_1)/std_feature_1
X_val[:, 1] = (X_val[:, 1] - mean_feature_2)/std_feature_2

X_test = test_data[['feature_1', 'feature_2']].values
y_test = test_data[['y']].values

X_test[:, 0] = (X_test[:, 0] - mean_feature_1)/std_feature_1
X_test[:, 1] = (X_test[:, 1] - mean_feature_2)/std_feature_2

print("Datasets normalized.")
print("Start training logistic regression model.")
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X, y)
print("Training is finished.")

y_pred = clf.predict(X)
average_precision = average_precision_score(y_true=y, y_score=y_pred)
print("Average precision score train set: " + str(average_precision))
plot_stats(y, y_pred, average_precision, figure_name='baseline_train.pdf')

y_pred = clf.predict(X_val)
average_precision = average_precision_score(y_true=y_val, y_score=y_pred)
print("Average precision score val set: " + str(average_precision))
plot_stats(y_val, y_pred, average_precision, figure_name='baseline_val.pdf')

y_pred = clf.predict(X_test)
average_precision = average_precision_score(y_true=y_test, y_score=y_pred)
print("Average precision score test set: " + str(average_precision))
plot_stats(y_test, y_pred, average_precision, figure_name='baseline_test.pdf')