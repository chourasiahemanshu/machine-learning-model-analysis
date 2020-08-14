import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from models.defaults import DEFAULTS
# TODO : Change the default values and test it
class GaussianPrRegressor():

    def __init__(self, dataset):
        self.dataset = dataset
        self.gpr = GaussianProcessRegressor(**DEFAULTS[dataset]['gaussian_pr']['defaults'])
        print("""
			**********************
			Gaussian Process Regressor
			**********************
		""")

    def train_and_predict(self, X, y, X_test):
        '''
        fit training dataset and predict values for test dataset
        '''
        self.gpr.fit(X, y)
        self.gpr.predict(X_test)

    def score(self, X, X_test, y, y_test):
        '''
        Returns the score of knn by fitting training data
        '''
        self.train_and_predict(X, y, X_test)
        return self.gpr.score(X_test, y_test)

    def create_new_instance(self, values):
        return GaussianProcessRegressor(**{**values})

    def param_grid(self, is_random=False):
        '''
        dictionary of hyper-parameters to get good values for each one of them
        '''
        # random search only accepts a dict for params whereas gridsearch can take either a dic or list of dict
        return DEFAULTS[self.dataset]['gaussian_pr']['param_grid']

    def get_sklearn_model_class(self):
        return self.gpr

    def __str__(self):
        return "GaussianProcessRegressor"