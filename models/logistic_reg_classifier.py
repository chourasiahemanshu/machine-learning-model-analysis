from sklearn.linear_model import LogisticRegression
from models.defaults import DEFAULTS
import numpy as np

class LogisticRegClassifier():

    def __init__(self,dataset):
        self.dataset = dataset
        self.logistic = LogisticRegression(**DEFAULTS[dataset]['logistic_reg']['defaults'])
        print("""
    		**********************
    		Logistic Regression 
    		**********************
    	""")

    def train_and_predict(self, X, y, X_test):
        '''
        fit training dataset and predict values for test dataset
        '''
        self.logistic.fit(X, y)
        self.logistic.predict(X_test)

    def score(self, X, X_test, y, y_test):
        '''
        Returns the score of logistic regression by fitting training data
        '''
        self.train_and_predict(X, y, X_test)
        return self.logistic.score(X_test, y_test)

    def create_new_instance(self, values):
        return LogisticRegression(**{**values})

    def param_grid(self, is_random=False):
        '''
        dictionary of hyper-parameters to get good values for each one of them
        '''
        # random search only accepts a dict for params whereas gridsearch can take either a dic or list of dict
        return DEFAULTS[self.dataset]['logistic_reg']['param_grid']

    def get_sklearn_model_class(self):
        return self.logistic

    def __str__(self):
        return "LogisticRegression"