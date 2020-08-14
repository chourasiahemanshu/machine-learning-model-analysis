from sklearn.ensemble import RandomForestRegressor
from models.defaults import DEFAULTS
import numpy as np

class RfRegressor():

    def __init__(self,dataset):
        self.dataset = dataset
        self.random_forest = RandomForestRegressor(**DEFAULTS[dataset]['rf']['defaults'],random_state=0)
        print("""
    		**********************
    		Random Forest Regression
    		**********************
    	""")

    def train_and_predict(self, X, y, X_test):
        '''
        fit training dataset and predict values for test dataset
        '''
        self.random_forest.fit(X, y)
        self.random_forest.predict(X_test)

    def score(self, X, X_test, y, y_test):
        '''
        Returns the score of Random Forest by fitting training data
        '''
        self.train_and_predict(X, y, X_test)
        return self.random_forest.score(X_test, y_test)

    def create_new_instance(self, values):
        return RandomForestRegressor(**{**values})

    def param_grid(self, is_random=False):
        '''
        dictionary of hyper-parameters to get good values for each one of them
        '''
        # random search only accepts a dict for params whereas gridsearch can take either a dic or list of dict
        return DEFAULTS[self.dataset]['rf']['param_grid']

    def get_sklearn_model_class(self):
        return self.random_forest

    def __str__(self):
        return "RandomForestRegression"