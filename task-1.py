#######################################################################################################
#LINEAR REGRESSION
#Basic numpy, pandas and matplotlib imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import metrics
from sklearn.linear_model import LinearRegression

#getting the directory
THIS_FOLDER = os.path.abspath('')


#setting the training and testing dataset path
class Dataset:
  train=os.path.join(THIS_FOLDER, 'wineQualityRed_train.csv')
  test=os.path.join(THIS_FOLDER, 'wineQualityRed_test.csv')

fields=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
fields1=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]

#X_TRAINING DATASET
train = pd.read_csv(Dataset.train, delimiter=';', header=None, skiprows=1,  names=fields)
#X_TESTING DATASET
test = pd.read_csv(Dataset.test, delimiter=';', header=None, skiprows=1,  names=fields)

print(train.head())
print(test.head())

############################################################################################################
##LINEAR REGRESSION FOR SINGLE ATTRIBUTE

 #considering only 'fixed acidity' attribute
x_train=train.iloc[:,0:1].values

y_train=train.iloc[:,-1].values
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#taking all test data for regression
x_test=test.iloc[:,0:1].values
y_test=test.iloc[:,-1].values

#predict y for test data
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.head())
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

#intercept
print("Intercept:", regressor.intercept_)
#slope
print("Slope:", regressor.coef_)

#plot the values for fixed acidity feature
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color='black')
plt.show()

########################################################################################################
##LINEAR REGRESSION FOR ALL ATTRIBUTES

x_train_all = np.array(train.drop(['quality'], axis=1))
y_train_all = np.array(train['quality'])

x_test_all = np.array(test.drop(['quality'], axis=1))
y_test_all = np.array(test['quality'])


regressor_all = LinearRegression()
regressor_all.fit(x_train_all,y_train_all)

#intercept
print("Intercept:", regressor_all.intercept_)
#slope
print("Slope:", regressor_all.coef_)

test_pred = regressor_all.predict(x_test_all)
print(test_pred[:10])

predicted_data = np.round_(test_pred)
print(predicted_data[:10])



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_all, test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_all, test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_all, test_pred)))