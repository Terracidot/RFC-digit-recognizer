"""https://www.kaggle.com/c/digit-recognizer/data"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

dataframe = pd.read_csv(r"C:\Users\Matthew\Google Drive\Python work\Kaggle Work\Digit recognizer data\train.csv")

print(dataframe.head())
print(dataframe.columns[0:10])

sns.set_style('darkgrid')
#sns.countplot(data=dataframe, x='Class')
#sns.pairplot(data=dataframe, hue='Class')

scaler = StandardScaler()
scaler.fit(dataframe.drop('label', axis=1))
scaled_features = scaler.fit_transform(dataframe.drop('label', axis=1))
print('scaled_features', scaled_features)

#scaled_data = scaler.transform(dataframe)
#print(scaled_data)

dataframe_features = pd.DataFrame(scaled_features, columns=dataframe.columns[1:])
print('dataframe features', dataframe_features)

#x = dataframe_features
#y = dataframe['label']
#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=64)

testdata = pd.read_csv(r"C:\Users\Matthew\Google Drive\Python work\Kaggle Work\Digit recognizer data\test.csv")
print('test data', testdata)

scalertest = StandardScaler()
scalertest.fit(testdata)
scaled_features_test = scalertest.fit_transform(testdata)
print(scaled_features_test)

dataframe_features_test = pd.DataFrame(scaled_features_test, columns=testdata.columns)
print('test dataframe features', dataframe_features_test)

x_train = dataframe_features
x_test = dataframe_features_test
y_train = dataframe['label']
y_test = dataframe_features_test
print('x_train', x_train)
print('x_test', x_test)
print('y_train', y_train)
print('y_test', y_test)


# rfc = RandomForestClassifier(n_estimators=100).fit(x_train, y_train)
# rfc_pred = rfc.predict(x_test)
#
# print(len(rfc_pred))
#
# d1 = pd.DataFrame(columns=['ImageId'])
# d2 = pd.DataFrame(rfc_pred, columns=['label'])
#
# # Join both datasets and print to a csv file
# d3 = pd.concat([d1, d2], axis=1)
# #pd.DataFrame(d3).to_csv('predictions.csv', index=['ImageId'])
# print(d3)
#
# #print('random forest matrix', confusion_matrix(y_test, rfc_pred))
# #print('random forest classification report', classification_report(y_test, rfc_pred))
# #print('MAE:', metrics.mean_absolute_error(y_test, rfc_pred))
# #print('MSE:', metrics.mean_squared_error(y_test, rfc_pred))  # Most used for real life situations NB!!!!
# #print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfc_pred)))
# #plt.tight_layout()
# #plt.show()
