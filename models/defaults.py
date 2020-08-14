import numpy as np
import random
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
# kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(5.0, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(5.0, 10.0)) + RBF(length_scale=2.0, length_scale_bounds=(5.0, 1.0))
DEFAULTS = {
	'./data/messidor_features.arff': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30),
				'leaf_size' : np.arange(1,50)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,1]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18), 
				'max_iter': 500, 
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))], 
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {


			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'class_weight': [{1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
				'solver': ['liblinear', 'saga'],
			}
		},
		'dt': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [20,30,40,50],
			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators" : [10,40,60],
				"max_depth" : [8, 15,20],
				"min_samples_split" : [ 15, 50,60],
				"min_samples_leaf" : [1, 2, 5],
			}
		},
		'ab': {
				'defaults': {
				# 'random_state' = 0, 'n_estimators' = 100,
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
 				'learning_rate' : [0.1,0.3,1],
 				# 'loss' : ['linear', 'exponential']
			}
		}
	},
	'./data/breast-cancer-wisconsin.data': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30),
				'leaf_size' : np.arange(1,50)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18), 
				'max_iter': 500, 
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))], 
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'class_weight': [{1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
				'solver': ['liblinear', 'saga'],
			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [3,5,10,15,20],
				"min_samples_leaf": [2,3,5],
				"max_leaf_nodes": [20,30,40,50],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
"n_estimators" : [10,40,60],
				"max_depth" : [8, 15,20],
				"min_samples_split" : [ 15, 50,60],
				"min_samples_leaf" : [1, 2, 5],
			}
		},
		'ab': {
				'defaults': {
				# 'random_state' = 0, 'n_estimators' = 100,
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
 				'learning_rate' : [0.1,0.3,1],
 				# 'loss' : ['linear', 'exponential']
			}
		}
	},
	'./data/statlog-australian-credit-data.data': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30),
				'leaf_size' : np.arange(1,50)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18), 
				'max_iter': 500, 
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))], 
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'solver': ['liblinear', 'saga'],

			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [3,2,4,5],
				"min_samples_leaf": [5,10,15,20,25],
				"max_leaf_nodes": [20,30,40],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [10, 40, 60],
				"max_depth": [8, 15, 20],
				"min_samples_split": [5,10,15],
				"min_samples_leaf": [1, 2, 5],

			}
		},
		'ab': {
				'defaults': {
				# 'random_state' = 0, 'n_estimators' = 100,
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
 				'learning_rate' : [0.1,0.3,1],
 				# 'loss' : ['linear', 'exponential']
			}
		}
	},
	'./data/statlog-german-credit-data.data': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30),
				'leaf_size' : np.arange(1,50)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18), 
				'max_iter': 500, 
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))], 
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'solver': ['liblinear', 'saga'],
			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [ 40, 60,70],
				"max_depth": [6,8, 15],
				"min_samples_split": [10,15, 20],
				"min_samples_leaf": [1, 2, 5],

			}
		},
		'ab': {
				'defaults': {
				# 'random_state' = 0, 'n_estimators' = 100,
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
 				'learning_rate' : [0.1,0.3,1],
 				# 'loss' : ['linear', 'exponential']
			}
		}
	},
	'./data/steel-plates-faults.NNA': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30),
				'leaf_size' : np.arange(1,50)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18), 
				'max_iter': 500, 
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))], 
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'solver': ['liblinear', 'saga'],


			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [3,5,6,7],
				"max_depth": [5,10,15],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [10, 60,70],
				"max_depth": [8, 15, 20],
				"min_samples_split": [10,15,20],
				"min_samples_leaf": [1, 2, 5],

			}
		},
		'ab': {
				'defaults': {
				# 'random_state' = 0, 'n_estimators' = 100,
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
 				'learning_rate' : [0.1,0.3,1],
 				# 'loss' : ['linear', 'exponential']
			}
		}
	},
	'./data/adult.data': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30),
				'leaf_size' : np.arange(1,50)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18), 
				'max_iter': 500, 
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))], 
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'solver': ['liblinear', 'saga'],


			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [10, 40, 60],
				"max_depth": [8, 15, 20],
				"min_samples_split": [15, 50, 60],
				"min_samples_leaf": [1, 2, 5],
			}
		},
		'ab': {
				'defaults': {
				# 'random_state' = 0, 'n_estimators' = 100,
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
 				'learning_rate' : [0.1,0.3,1],
 				# 'loss' : ['linear', 'exponential']
			}
		}
	},
	'./data/seismic-bumps.arff': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30),
				'leaf_size' : np.arange(1,50)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18),
				'max_iter': 500,
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))],
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'solver': ['liblinear', 'saga'],


			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [10, 40, 60],
				"max_depth": [8, 15, 20],
				"min_samples_split": [15, 50, 60],
				"min_samples_leaf": [1, 2, 5],

			}
		},
		'ab': {
				'defaults': {
				# 'random_state' = 0, 'n_estimators' = 100,
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
 				'learning_rate' : [0.1,0.3,1],
 				# 'loss' : ['linear', 'exponential']
			}
		}
	},
	'./data/yeast.data': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30),
				'leaf_size': np.arange(1, 50)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18),
				'max_iter': 500,
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))],
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'solver': ['liblinear', 'saga'],


			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [10, 40, 60],
				"max_depth": [8, 15, 20],
				"min_samples_split": [15, 50, 60],
				"min_samples_leaf": [1, 2, 5],

			}
		},
		'ab': {
				'defaults': {
				# 'random_state' = 0, 'n_estimators' = 100,
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
 				'learning_rate' : [0.1,0.3,1],
 				# 'loss' : ['linear', 'exponential']
			}
		}
	},
	'./data/default_of_credit_card_clients.xls': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30),
				'leaf_size': np.arange(1, 50)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18),
				'max_iter': 500,
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))],
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'class_weight': [{1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
				'solver': ['liblinear', 'saga'],


			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [10, 40, 60],
				"max_depth": [8, 15, 20],
				"min_samples_split": [15, 50, 60],
				"min_samples_leaf": [1, 2, 5]
			}
		},
		'ab': {
				'defaults': {
				# 'random_state' = 0, 'n_estimators' = 100,
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {

			}
		}
	},
	'./data/ThoraricSurgery.arff': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30),
				'leaf_size': np.arange(1, 50)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18),
				'max_iter': 500,
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))],
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'class_weight': [{1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
				'solver': ['liblinear', 'saga'],


			}
		},
		'dt': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [10, 40, 60],
				"max_depth": [8, 15, 20],
				"min_samples_split": [15, 50, 60],
				"min_samples_leaf": [1, 2, 5],

			}
		},
		'ab': {
				'defaults': {
				# 'random_state' = 0, 'n_estimators' = 100,
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
 				'learning_rate' : [0.1,0.3,1],
 				# 'loss' : ['linear', 'exponential']
			}
		}
	},
	'./data_regression/bike_sharing_hour.csv': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept':[True, False],
				'normalize':[True, False],
				'copy_X':[True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
				'random_state': [0],
              	"min_samples_split": [10, 15, 20],
              	"max_depth": [2, 6, 8],
              	"min_samples_leaf": [20, 40, 100],
              	"max_leaf_nodes": [50,100],
			}
		},
		'svr': {
			'defaults': {
				'C':100,
				'gamma':0.1,
				'epsilon':.1
			},
			'param_grid': {
                'C': [0.1,1, 10],
			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {
				"alpha":[0.1,0.2,0.3,0.4],
				"normalize_y":[True, False]
			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [40, 60, 70],
				"max_depth": [6, 8, 15],
				"min_samples_split": [10, 15, 20],
				"min_samples_leaf": [1, 2, 5],

			}
		},
		'nn': {
			'defaults': {
				'max_iter':10
			},
			'param_grid': {
				'hidden_layer_sizes': [(50,50,50), (100,)],
				'activation': ['tanh', 'relu'],
				'alpha': [ 0.05],
			}
		},
		'ab': {
			'defaults': {
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
				'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
				'loss': ['linear', 'square', 'exponential']
			}
		}
	},
	'./data_regression/dataset_Facebook.csv': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
'random_state': [0],
				"min_samples_split": [10, 15, 20, 40],
				"max_depth": [4,6,8, 10],
				"min_samples_leaf": [10,20, 40],
				"max_leaf_nodes": [40,50, 100],
			}
		},
		'svr': {
			'defaults': {
				'C':100,
				'gamma':0.1,
				'epsilon':.1
			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {
				"alpha":[0.1,0.2,0.3,0.4],
				"normalize_y":[True, False]
			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [40, 60, 70],
				"max_depth": [6, 8, 15],
				"min_samples_split": [10, 15, 20],
				"min_samples_leaf": [1, 2, 5],

			}
		},
		'nn': {
			'defaults': {
				'max_iter':500
			},
			'param_grid': {
				'hidden_layer_sizes': [ (100,)],
				'activation': ['tanh' ],
				'solver': ['sgd', 'adam'],
				'alpha': [ 0.05],
				'learning_rate': ['constant']
			}
		},
		'ab': {
			'defaults': {
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
				'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
				'loss': ['linear', 'square', 'exponential']
			}
		}
	},
	'./data_regression/qsar_aquatic_toxicity.csv': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
				'random_state': [0],
				"min_samples_split": [10, 15, 20, 40],
				"max_depth": [4,6,8, 10],
				"min_samples_leaf": [10,20, 40],
				"max_leaf_nodes": [40,50, 100],
			}
		},
		'svr': {
			'defaults': {
				'C':100,
				'gamma':0.1,
				'epsilon':.1
			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {
				"alpha":[0.1,0.2,0.3,0.4],
				"normalize_y":[True, False]
			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [40, 60, 70],
				"max_depth": [6, 8, 15],
				"min_samples_split": [10, 15, 20],
				"min_samples_leaf": [1, 2, 5],

			}
		},
		'nn': {
			'defaults': {
				'max_iter':200
			},
			'param_grid': {
				'hidden_layer_sizes': [ (100,)],
				'activation': ['tanh', 'relu'],
				'solver': ['sgd', 'adam'],
				'alpha': [ 0.05],
				'learning_rate': ['constant','adaptive']
			}
		},
		'ab': {
			'defaults': {
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
				'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
				'loss': ['linear', 'square', 'exponential']
			}
		}
	},
	'./data_regression/sgemm_product.csv': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
                'random_state': [0]
			}
		},
		'svr': {
			'defaults': {
				'C':100,
				'gamma':0.1,
				'epsilon':.1
			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [40, 60, 70],
				"max_depth": [6, 8, 15],
				"min_samples_split": [10, 15, 20],
				"min_samples_leaf": [1, 2, 5],

			}
		},
		'nn': {
			'defaults': {
				'max_iter':5
			},
			'param_grid': {
				# 'hidden_layer_sizes': [ (100,)],
				# # 'activation': ['tanh', 'relu'],
				# 'solver': ['sgd'],
				# 'alpha': [ 0.5],
			}
		},
		'ab': {
			'defaults': {
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
				'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
				'loss': ['linear', 'square', 'exponential']
			}
		}
	},
	'./data_regression/Concrete_Data.xls': {
		'linear_reg': {
			'defaults': {

			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],
			}
		},
		'svr': {
			'defaults': {
				'C':100,
				'gamma':0.1,
				'epsilon':.1
			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {
				"alpha":[0.1,0.2,0.3,0.4],
				"normalize_y":[True, False]
			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [40, 60, 70],
				"max_depth": [6, 8, 15],
				"min_samples_split": [10, 15, 20],
				"min_samples_leaf": [1, 2, 5],

			}
		},
		'nn': {
			'defaults': {
				'max_iter':20
			},
			'param_grid': {
				'hidden_layer_sizes': [ (100,)],
				'activation': ['tanh', 'relu'],
				'solver': ['sgd', 'adam'],
				'alpha': [0.05],
			}
		},
		'ab': {
			'defaults': {
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
				'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
				'loss': ['linear', 'square', 'exponential']
			}
		}
	},
	'./data_regression/winequality-red.csv': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
				'random_state': [0],
				'max_depth':[None, 5, 3, 1],
                'min_samples_split':np.linspace(0.1, 1.0, 5, endpoint=True),
                'min_samples_leaf':np.linspace(0.1, 0.5, 5, endpoint=True),
                'max_features':['auto', 'sqrt', 'log2'],
			}
		},
		'svr': {
			'defaults': {
				'C':100,
				'gamma':0.1,
				'epsilon':.1
			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {
				"alpha":[0.1,0.2,0.3,0.4],
				"normalize_y":[True, False]
			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [70,80],
				"max_depth": [6, 15,20],
				"min_samples_split": [5,10, 15],
				"min_samples_leaf": [1, 2, 3],

			}
		},
		'nn': {
			'defaults': {
				'max_iter': 20
			},
			'param_grid': {
				'hidden_layer_sizes': [(100,)],
				'activation': ['tanh'],
				'solver': ['sgd', 'adam'],
				'alpha': [0.05],
				'learning_rate': ['constant','adaptive']
			}
		},
		'ab': {
			'defaults': {
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
				'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
				'loss': ['linear', 'square', 'exponential']
			}
		}
	},
	'./data_regression/winequality-white.csv': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
				'random_state': [0],
				'max_depth':[None, 5, 3, 1],
                'min_samples_split':np.linspace(0.1, 1.0, 5, endpoint=True),
                'min_samples_leaf':np.linspace(0.1, 0.5, 5, endpoint=True),
                'max_features':['auto', 'sqrt', 'log2'],
			}
		},
		'svr': {
			'defaults': {
				'C':100,
				'gamma':0.1,
				'epsilon':.1
			},
			'param_grid': {
                'C': [1, 10, 100],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {
				"alpha":[0.1,0.2,0.3,0.4],
				"normalize_y":[True, False]
			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [60,70,80],
				"max_depth": [6, 15,20],
				"min_samples_split": [5,10, 12],
				"min_samples_leaf": [1, 2, 3],

			}
		},
		'nn': {
			'defaults': {
				'max_iter':10
			},
			'param_grid': {
				'hidden_layer_sizes': [(100,)],
				'activation': ['tanh', 'relu'],
				'solver': ['sgd'],
				'alpha': [ 0.05],
				'learning_rate': ['constant']
			}
		},
		'ab': {
			'defaults': {
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
				'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
				'loss': ['linear', 'square', 'exponential']
			}
		}
	},
	'./cfar10/data_batch': {
		'dt': {
			'defaults': {
				'max_depth': 10000, 'random_state': 0, 'min_samples_split': 5,
			},
			'param_grid': {
				'max_depth': [100, 500, 1000, 10000], 'max_features': ['auto', 'log2', None], 'criterion': ['gini', 'entropy'],
			}
		},
	},
  	'./data_regression/student-por.csv': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
                'random_state': [0],
                'max_depth':np.linspace(1, 32, 16, endpoint=True),
                'min_samples_split':np.linspace(0.1, 1.0, 5, endpoint=True),
                'min_samples_leaf':np.linspace(0.1, 0.5, 5, endpoint=True),
                # 'max_features':list(range(1,train.shape[1])),
			}
		},
		'svr': {
			'defaults': {
				'C':100,
				'gamma':0.1,
				'epsilon':.1
			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {
				"alpha":[0.1,0.2,0.3,0.4],
				"normalize_y":[True, False]
			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
                'max_depth': [ 40,  70,],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [ 40, 60, 70]}
		},
		'nn': {
			'defaults': {
				'max_iter':100
			},
			'param_grid': {
				'hidden_layer_sizes': [ (100,)],
				'activation': ['tanh', 'relu'],
				'solver': ['sgd', 'adam'],
				'alpha': [0.1, 0.05],
				'learning_rate': ['constant','adaptive']
			}
		},
		'ab': {
			'defaults': {
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
				'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
				'loss': ['linear', 'square', 'exponential']
			}
		}
	},
	'./data_regression/communities.data': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
                'random_state': [0],
				"min_samples_split": [5,10],
				"max_depth": [8, 10,14],
				"min_samples_leaf": [10,20, 40],
				"max_leaf_nodes": [10,20,40],
			}
		},
		'svr': {
			'defaults': {

			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {
				"alpha":[0.1,0.2,0.3,0.4],
				"normalize_y":[True, False]
			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [40, 60, 70],
				"max_depth": [6, 8, 15],
				"min_samples_split": [60,80,30],
				"min_samples_leaf": [1, 2, 5],

			}
		},
		'nn': {
			'defaults': {
				'max_iter':10
			},
			'param_grid': {
				'hidden_layer_sizes': [(100,)],
				'activation': [ 'relu'],
				# 'solver': ['sgd', 'adam'],
				'alpha': [ 0.05,0.1],
				'learning_rate': ['constant','adaptive']
			}
		},
		'ab': {
			'defaults': {
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
				'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
				'loss': ['linear', 'square', 'exponential']
			}
		}
	},
	'./data_regression/ACT2_competition_training.npz': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
                'random_state': [0],
				"min_samples_split": [4,5,6],
				"max_depth": [8, 10,14],
				"min_samples_leaf": [10,20, 40],
				"max_leaf_nodes": [40,50],
			}
		},
		'svr': {
			'defaults': {

			},
			'param_grid': {
                'C': [0.1, 1, 10],
			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {
				"alpha":[0.1,0.2,0.3,0.4],
				"normalize_y":[True, False]
			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'nn': {
			'defaults': {
				'max_iter':100
			},
			'param_grid': {
				'hidden_layer_sizes': [(100,)],
				'activation': ['tanh', 'relu'],
				'solver': ['sgd', 'adam'],
				'alpha': [0.10, 0.05],
				'learning_rate': ['constant','adaptive']
			}
		},
		'ab': {
			'defaults': {
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
				'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
				'loss': ['linear', 'square', 'exponential']
			}
		}
	},
	'./data_regression/ACT4_competition_training.npz': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
                'random_state': [0],
				"min_samples_split": [3,4,5],
				"max_depth": [ 10,14,16],
				"min_samples_leaf": [10,20, 40],
				"max_leaf_nodes": [50,60],
			}
		},
		'svr': {
			'defaults': {

			},
			'param_grid': {
                'C': [0.1,1, 10],
                'kernel': ['rbf','sigmoid'],
                'gamma': ['auto','scale']

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {
				"alpha":[0.1,0.2,0.3,0.4],
				"normalize_y":[True, False]
			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [40, 60, 70],
				"max_depth": [6, 8, 15],
				"min_samples_split": [10, 15, 20],
				"min_samples_leaf": [1, 2, 5],

			}
		},
		'nn': {
			'defaults': {
				'max_iter':100
			},
			'param_grid': {
				'hidden_layer_sizes': [ (100,)],
				'activation': ['tanh', 'relu'],
				'solver': ['sgd', 'adam'],
				'alpha': [0.10, 0.05],
				'learning_rate': ['constant','adaptive']
			}
		},
		'ab': {
			'defaults': {
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
				'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
				'loss': ['linear', 'square', 'exponential']
			}
		}
	},
	'./data_regression/parkinson_train_data.txt': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
 'random_state': [0],
				"min_samples_split": [2,3,4],
				"max_depth": [3,4,5],
				"min_samples_leaf": [40,60,80],
				"max_leaf_nodes": [10,20,30],
			}
		},
		'svr': {
			'defaults': {

			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {
				"alpha":[0.1,0.2,0.3,0.4],
				"normalize_y":[True, False]
			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {
				"n_estimators": [40, 60, 70],
				"max_depth": [6, 8, 15],
				"min_samples_split": [10, 15, 20],
				"min_samples_leaf": [1, 2, 5],

			}
		},
		'nn': {
			'defaults': {
				'max_iter':100
			},
			'param_grid': {
				'hidden_layer_sizes': [ (100,)],
				'activation': ['tanh', 'relu'],
				'solver': ['sgd', 'adam'],
				'alpha': [0.1, 0.05],
				'learning_rate': ['constant','adaptive']
			}
		},
		'ab': {
			'defaults': {
				# 'random_state' = 0, 'n_estimators' = 100,
				'random_state': 0, 'n_estimators': 100,
			},
			'param_grid': {
				'n_estimators': [50, 100],
 				'learning_rate' : [0.01,0.05,0.1,0.3,1],
 				'loss' : ['linear', 'square', 'exponential']
			}
		}
	}
}