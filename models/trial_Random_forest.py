import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import sklearn.tree        # For DecisionTreeClassifier class
import sklearn.ensemble    # For RandomForestClassifier class
import sklearn.datasets    # For make_circles
import sklearn.metrics     # For accuracy_score
# from sklearn.ensemble import RandomForestClassfier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split # Import train_test_split function


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

sel = SelectFromModel(sklearn.ensemble.RandomForestClassifier(n_estimators = 100))
sel.fit(X_train, y_train)
sel.get_support()
selected_feat= X_train.columns[(sel.get_support())]
len(selected_feat)
print(selected_feat)

reg = sklearn.ensemble.RandomForestClassifier(n_estimators = 100)
reg.fit(X_train, y_train)

print('Accuracy of RF classifier on training set: {:.2f}'
     .format(reg.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
     .format(reg.score(X_test, y_test)))

# importances = reg.feature_importances_
# indices = np.argsort(importances)
#
# print("Graph ")
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importances[indices], color='#8f63f4', align='center')
# plt.yticks(range(len(indices)), feature_cols[indices])
# plt.xlabel('Relative Importance')
# plt.show()

# y_pred = reg.predict(X_test)


# pd.series(sel.estimator_,selected_feat,.ravel()).hist()