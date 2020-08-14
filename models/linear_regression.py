from sklearn.linear_model import LinearRegression
from models.defaults import DEFAULTS
from sklearn.metrics import mean_squared_error


class LinearReg():

    def __init__(self, dataset):
        self.dataset = dataset
        self.linear = LinearRegression(**DEFAULTS[dataset]['linear_reg']['defaults'])
        print("""
        **********************
        Linear Regression
        **********************
        """)

    def train_and_predict(self, X, y, X_test):
        '''
        fit training dataset and predict values for test dataset
        '''
        self.linear.fit(X, y)
        self.linear.predict(X_test)

    def score(self, X, X_test, y, y_test):
        '''
        Returns the score of Linear Regression by fitting training data
        '''
        self.train_and_predict(X, y, X_test)
        return self.linear.score(X_test, y_test)
        # self.linear.fit(X, y)
        # y_pred = self.linear.predict(X_test)
        # return mean_squared_error(y_test, y_pred)

    def create_new_instance(self, values):
        return LinearRegression(**values)

    def param_grid(self, is_random=False):
        '''
        dictionary of hyper-parameters to get good values for each one of them
        '''
        return DEFAULTS[self.dataset]['linear_reg']['param_grid']

    def get_sklearn_model_class(self):
        return self.linear

    def __str__(self):
        return "Linear_Regression"
