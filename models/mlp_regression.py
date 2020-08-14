from sklearn.neural_network import MLPRegressor
import numpy as np

#https://stackoverflow.com/questions/41069905/trouble-fitting-simple-data-with-mlpregressor

# x = np.arange(0.0, 1, 0.01).reshape(-1, 1)
# y = np.sin(2 * np.pi * x).ravel()
#
# nn = MLPRegressor(
#     hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
#     learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
#     random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#
# n = nn.fit(x, y)
# test_x = np.arange(0.0, 1, 0.05).reshape(-1, 1)
# test_y = nn.predict(test_x)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.scatter(x, y, s=1, c='b', marker="s", label='real')
# ax1.scatter(test_x,test_y, s=10, c='r', marker="o", label='NN Prediction')
# plt.show()
from models.defaults import DEFAULTS


class MlpRegression():

    def __init__(self, dataset):
        self.dataset = dataset
        self.nn = MLPRegressor(**DEFAULTS[dataset]['nn']['defaults'])
        print("""
			**********************
			Neural Network Regression
			**********************
		""")

    def train_and_predict(self, X, y, X_test):
        '''
        fit training dataset and predict values for test dataset
        '''
        self.nn.fit(X, y)
        self.nn.predict(X_test)

    def score(self, X, X_test, y, y_test):
        '''
        Returns the score of knn by fitting training data
        '''
        self.train_and_predict(X, y, X_test)
        return self.nn.score(X_test, y_test)

    def create_new_instance(self, values):
        return MLPRegressor(**{**values, 'random_state': 0})

    def param_grid(self, is_random=False):
        '''
        dictionary of hyper-parameters to get good values for each one of them
        '''
        return DEFAULTS[self.dataset]['nn']['param_grid']

    def get_sklearn_model_class(self):
        return self.nn

    def __str__(self):
        return "Neural Network Regression"