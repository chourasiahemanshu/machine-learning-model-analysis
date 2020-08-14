from sklearn.ensemble import AdaBoostClassifier
from models.defaults import DEFAULTS
import numpy as np

class ABClassifier():

    def __init__(self,dataset):
        self.dataset = dataset
        self.adaboost = AdaBoostClassifier(**DEFAULTS[dataset]['ab']['defaults'])
        print("""
    		**********************
    		Ada Boost Classifier
    		**********************
    	""")

    def train_and_predict(self, X, y, X_test):
        '''
        fit training dataset and predict values for test dataset
        '''
        self.adaboost.fit(X, y)
        self.adaboost.predict(X_test)

    def score(self, X, X_test, y, y_test):
        '''
        Returns the score of Ada Boost by fitting training data
        '''
        self.train_and_predict(X, y, X_test)
        return self.adaboost.score(X_test, y_test)

    def create_new_instance(self, values):
        return AdaBoostClassifier(**{**values})

    def param_grid(self, is_random=False):
        '''
        dictionary of hyper-parameters to get good values for each one of them
        '''
        # random search only accepts a dict for params whereas gridsearch can take either a dic or list of dict
        return DEFAULTS[self.dataset]['ab']['param_grid']

    def get_sklearn_model_class(self):
        return self.adaboost

    def __str__(self):
        return "AdaBoost"