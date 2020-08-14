from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import pandas as pd


col_names = ['ID','Amount of the given credit', 'Gender', 'Education', 'Marital status', 'Age', 'repayment status in September, 2005', 'repayment status in August, 2005', 'repayment status in July, 2005', 'repayment status in June, 2005','repayment status in May, 2005','repayment status in April, 2005','Amount of Bill Statement in September, 2005', 'Amount of Bill Statement in August, 2005', 'Amount of Bill Statement in July, 2005', 'Amount of Bill Statement in June, 2005','Amount of Bill Statement in May, 2005','Amount of Bill Statement in April, 2005','Amount Payed in September, 2005', 'Amount Payed in August, 2005', 'Amount Payed in July, 2005', 'Amount Payed in June, 2005','Amount Payed in May, 2005','Amount Payed in April, 2005','default_payment']
# data = pd.read_excel (r'./../data/try.xls', header=None,names=col_names)
data = pd.read_excel (r'./../data/default_of_credit_card_clients.xls', header=None,names=col_names)
# print (data)
data= data.iloc[2:]
del data['ID']
feature_cols = ['Amount of the given credit', 'Gender', 'Education', 'Marital status', 'Age', 'repayment status in September, 2005', 'repayment status in August, 2005', 'repayment status in July, 2005', 'repayment status in June, 2005','repayment status in May, 2005','repayment status in April, 2005','Amount of Bill Statement in September, 2005', 'Amount of Bill Statement in August, 2005', 'Amount of Bill Statement in July, 2005', 'Amount of Bill Statement in June, 2005','Amount of Bill Statement in May, 2005','Amount of Bill Statement in April, 2005','Amount Payed in September, 2005', 'Amount Payed in August, 2005', 'Amount Payed in July, 2005', 'Amount Payed in June, 2005','Amount Payed in May, 2005','Amount Payed in April, 2005']
X = data[feature_cols] # Features
y = data.default_payment # Target variable
# y= y.iloc[2:]
y=y.astype('int')
# print(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Accuracy: 0.8888888888888888

#
#--------------------------------------------------------------- changing learner to SVC ---- any can be used
# # Load libraries
# from sklearn.ensemble import AdaBoostClassifier
#
# # Import Support Vector Classifier
# from sklearn.svm import SVC
# #Import scikit-learn metrics module for accuracy calculation
# from sklearn import metrics
# svc=SVC(probability=True, kernel='linear')
#
# # Create adaboost classifer object
# abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)
#
# # Train Adaboost Classifer
# model = abc.fit(X_train, y_train)
#
# #Predict the response for test dataset
# y_pred = model.predict(X_test)