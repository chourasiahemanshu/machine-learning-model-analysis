import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz


# conda install -c anaconda graphviz
# conda install -c conda-forge pydotplus

import conda.cli.python_api as Conda
import sys
import os

try:
    import graphviz
except:
    print ("Graphiz not found, Installing Grpahiz ")
    os.system("conda install -c anaconda graphviz")
    import graphviz

try:
    import pydotplus
except:
    print("pydotplus not found, Installing pydotplus ")
    os.system("conda install -c  conda-forge pydotplus")
    import pydotplus


    # print("pydotplus not fount.... installing pydotplus using Conda and updating conda packages : conda install -c conda-forge pydotplus")
    # (stdout_str, stderr_str, return_code_int) = Conda.run_command(
    #     Conda.Commands.INSTALL,
    #     '-c', 'conda-forge',
    #     'pydotplus',
    # use_exception_handler = True, stdout = sys.stdout, stderr = sys.stderr
    # )

# try:
#     import graphviz
# except:
#     print("graphviz not fount.... installing graphviz using Conda and updating conda packages : conda install -c anaconda graphviz")
#     (stdout_str, stderr_str, return_code_int) = Conda.run_command(
#         Conda.Commands.INSTALL,
#         '-c', 'anaconda',
#         'graphviz',
#         use_exception_handler=True, stdout=sys.stdout, stderr=sys.stderr
#     )

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


clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


print("Drawing GRaph")
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('credit.png')
Image(graph.create_png())



# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())