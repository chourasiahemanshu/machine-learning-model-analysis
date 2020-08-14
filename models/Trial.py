import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For Reference
# https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python

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
print(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

print("------------- Test Data --------------")
print(X_test)
print("------------- Test Data --------------")
print(y_test)


logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)




cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
# Here, you can see the confusion matrix in the form of the array object.
# The dimension of this matrix is 2*2 because this model is binary classification.
# You have two classes 0 and 1. Diagonal values represent accurate predictions, while non-diagonal elements are inaccurate predictions.
# In the output, 119 and 36 are actual predictions, and 26 and 11 are incorrect predictions.




class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


#
# ROC Curve
# Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the false positive rate.
# It shows the tradeoff between sensitivity and specificity.

y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()