# from sklearn.svm import SVC
from models.defaults import DEFAULTS
from sklearn.svm import SVR

# TODO : Check THis file for SVR

class SvrRegression():

    def __init__(self, dataset):
        self.dataset = dataset
        self.svr = SVR(**{**DEFAULTS[dataset]['svr']['defaults']})
        print("""
			**********************
			    SVR Regression
			**********************
		""")

    def train_and_predict(self, X, y, X_test):
        '''
        fit training dataset and predict values for test dataset
        '''
        self.svr.fit(X, y)
        self.svr.predict(X_test)

    def score(self, X, X_test, y, y_test):
        '''
        Returns the score of knn by fitting training data
        '''
        self.train_and_predict(X, y, X_test)
        return self.svr.score(X_test, y_test)

    def create_new_instance(self, values):
        return SVR(**{**values})

    def param_grid(self, is_random=False):
        '''
        dictionary of hyper-parameters to get good values for each one of them
        '''
        return DEFAULTS[self.dataset]['svr']['param_grid']


    def get_sklearn_model_class(self):
        return self.svr

    def __str__(self):
        return "SVR"