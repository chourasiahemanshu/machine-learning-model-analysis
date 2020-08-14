from sklearn.neighbors import KNeighborsClassifier
from models.defaults import DEFAULTS

class KnnClassifier():
	
	def __init__(self, dataset):
		self.dataset = dataset
		self.knn = KNeighborsClassifier(**DEFAULTS[dataset]['knn']['defaults'])
		print("""
			**********************
			KNN
			**********************
		""")
	
	def train_and_predict(self, X, y, X_test):
		'''
		fit training dataset and predict values for test dataset 
		'''
		self.knn.fit(X,y)
		self.knn.predict(X_test)

	def score(self, X, X_test, y, y_test):
		'''
		Returns the score of knn by fitting training data
		'''
		self.train_and_predict(X, y, X_test)
		return self.knn.score(X_test, y_test)

	def create_new_instance(self, values):
		return KNeighborsClassifier(**values)

	def param_grid(self, is_random=False):
		'''
		dictionary of hyper-parameters to get good values for each one of them
		'''
		return DEFAULTS[self.dataset]['knn']['param_grid']

	def get_sklearn_model_class(self):
		return self.knn

	def __str__(self):
		return "KNN"