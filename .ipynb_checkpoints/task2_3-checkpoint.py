##Linear Regression Classifier

#importing all the packages
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


import os

#getting the directory
THIS_FOLDER = os.path.abspath('')


#setting the training and testing dataset path
class Dataset:
  train=os.path.join(THIS_FOLDER, 'data/wineQualityRed_train.csv')
  test=os.path.join(THIS_FOLDER, 'data/wineQualityRed_test.csv')

#declaring features
fields=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

#X_TRAINING DATASET
train_wine = pd.read_csv(Dataset.train, delimiter=';', header=None, skiprows=1,  names=fields)

print("X_TRAINING DATA:")
print(train_wine.head(5))

print("***************************************************************************************")
# CLASSIFING 0-bad AND 1-good and forming Y_TRAINING DATASET
quality_train = train_wine["quality"].apply(lambda  q:0 if q<7 else 1)

print("Y_TRAINING DATA:")
print(quality_train.head(5))


#similarly for testing data

#X_TESTING DATASET
test_wine = pd.read_csv(Dataset.test, delimiter=';', header=None, skiprows=1,  names=fields)
print("X_TESTING DATA:")
print(test_wine.head(5))

print("***************************************************************************************") 

#Y_TESTING DATASET
quality_test = test_wine["quality"].apply(lambda  q:0 if q<7 else 1)
print("Y_TESTING DATA:")
print(quality_test.head(10))


#declaring linear regression model 

print("********************LINEAR REGRESSION CLASSIFIER***************************")
l_r= LinearRegression()

# training the model with X_TRAINING AND Y_TRAINING
l_r.fit(train_wine, quality_train)

#Getting Y_PREDICTION
pred1=l_r.predict(test_wine)

print("Y_PREDICTION DATA:(first 10 values)")
print(pred1[:10])

#MEAN OF THE Y_PREDICTION
mid = np.mean(pred1)

print("mean: ", mid)

pred= []


#Classifing y_prediction based on the mean value
for p in pred1 :
    if p < mid:
        pred.append(0)
    elif p > mid:
        pred.append(1)
        
print("Y_PREDICTION DATA after classificaion:(first 10 values)")
print(pred[:10])
        
#Getting the confusion matrix
cm1 = confusion_matrix(quality_test,pred)
print('Confusion Matrix : \n', cm1)
#prints [[TP,FN],[FP,TN]]

total1=sum(sum(cm1))


accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

precision1=(cm1[0,0])/(cm1[0,0]+cm1[1,0])
print ('Precision : ', precision1)


recall1=(cm1[0,0])/(cm1[0,0]+cm1[1,1])
print ('Recall : ', recall1) 

f_measure=(2*recall1*precision1)/(precision1+recall1)
print('f measure: ', f_measure)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)

##Logistic Regression Classifier

print("********************LOGISTIC REGRESSION CLASSIFIER***************************")
#declaring logistic regression model 
logisticRegr = LogisticRegression()

#training the model with X_TRAINING AND Y_TRAINING
logisticRegr.fit(train_wine, quality_train)

#Y_PREDICTION
pred=logisticRegr.predict(test_wine)

print("Y_PREDICTION:(first 10 values)")
print(pred[:10])

cm1 = confusion_matrix(quality_test,pred)
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

precision1=(cm1[0,0])/(cm1[0,0]+cm1[1,0])
print ('Precision : ', precision1)


recall1=(cm1[0,0])/(cm1[0,0]+cm1[1,1])
print ('Recall : ', recall1) 

f_measure=(2*recall1*precision1)/(precision1+recall1)
print('f measure: ', f_measure)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)

##SVM Classifier
print("********************SVM CLASSIFIER***************************")

#declaring svm model 
cls=svm.SVC(kernel="linear")

#training the model with X_TRAINING AND Y_TRAINING
cls.fit(train_wine,quality_train)

#Y_PREDICTION
pred=cls.predict(test_wine)


print("Y_PREDICTION:(first 10 values)")
print(pred[:10])

cm1 = confusion_matrix(quality_test,pred)
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

precision1=(cm1[0,0])/(cm1[0,0]+cm1[1,0])
print ('Precision : ', precision1)


recall1=(cm1[0,0])/(cm1[0,0]+cm1[1,1])
print ('Recall : ', recall1) 

f_measure=(2*recall1*precision1)/(precision1+recall1)
print('f measure: ', f_measure)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)


##Naive Bayes Classifier

print("********************NAIVE BAYES CLASSIFIER***************************")
#declaring Naive Bayes model
BernNB= BernoulliNB(binarize=True)

#training the model with X_TRAINING AND Y_TRAINING
BernNB.fit(train_wine, quality_train)
print(BernNB)
#Y_PREDICTION
y_pred=BernNB.predict(test_wine)


print("Y_PREDICTION:(first 10 values)")
print(y_pred[:10])


cm1 = confusion_matrix(quality_test,y_pred)
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

precision1=(cm1[0,0])/(cm1[0,0]+cm1[1,0])
print ('Precision : ', precision1)


recall1=(cm1[0,0])/(cm1[0,0]+cm1[1,1])
print ('Recall : ', recall1) 

f_measure=(2*recall1*precision1)/(precision1+recall1)
print('f measure: ', f_measure)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)