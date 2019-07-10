import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

train_data = pd.read_csv('dataset/Item_pairs_features_train.csv')
val_data = pd.read_csv('dataset/Item_pairs_features_val.csv')

X = train_data[['feature_1', 'feature_2']].values
y = train_data[['y']].values

X_val = val_data[['feature_1', 'feature_2']].values
y_val = val_data[['y']].values

print(X.shape)
print(y.shape)

clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X, y)

y_pred = clf.predict(X)
average_precision = average_precision_score(y_true=y, y_score=y_pred)
print("Averege precision score train set: " + str(average_precision))

y_pred = clf.predict(X_val)
average_precision = average_precision_score(y_true=y_val, y_score=y_pred)
print("Averege precision score val set: " + str(average_precision))