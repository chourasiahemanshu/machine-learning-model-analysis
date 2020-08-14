from sklearn.svm import SVC
from models.defaults import DEFAULTS

class SvmClassifier():
	
	def __init__(self, dataset):
		self.dataset = dataset
		self.svm = SVC(**{**DEFAULTS[dataset]['svm']['defaults'], 'probability': True})
		print("""
			**********************
			SVM
			**********************
		""")
	
	def train_and_predict(self, X, y, X_test):
		'''
		fit training dataset and predict values for test dataset 
		'''
		self.svm.fit(X,y)
		self.svm.predict(X_test)

	def score(self, X, X_test, y, y_test):
		'''
		Returns the score of knn by fitting training data
		'''
		self.train_and_predict(X, y, X_test)
		return self.svm.score(X_test, y_test)

	def create_new_instance(self, values):
		return SVC(**{**values, 'random_state': 0, 'probability': True})

	def param_grid(self, is_random=False):
		'''
		dictionary of hyper-parameters to get good values for each one of them
		'''
		return DEFAULTS[self.dataset]['svm']['param_grid']

	def get_sklearn_model_class(self):
		return self.svm

	def __str__(self):
		return "SVM"