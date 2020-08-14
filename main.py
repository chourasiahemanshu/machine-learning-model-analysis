from models.model import Model
from models.knn_classifier import KnnClassifier
from models.svm_classifier import SvmClassifier
from models.gaussian_nb_classifier import GaussianNbClassifier
from models.mlp_classifier import MlpClassifier
from models.logistic_reg_classifier import LogisticRegClassifier
from models.decision_tree_classifier import DTClassifier
from models.random_forest_classifier import RfClassifier
from models.ada_boost_classifier import ABClassifier
from models.gaussian_process_regressor import GaussianPrRegressor
from models.svr_regression import SvrRegression
from models.linear_regression import LinearReg
from models.decision_tree_regression import DTRegression
from models.random_forest_regressor import RfRegressor
from models.mlp_regression import MlpRegression
from models.ada_boost_regressor import ABRegressor
from models.cifar_models import CifarModel
import warnings

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignore sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)       # Ignore sklearn deprecation warnings

@ignore_warnings(category=ConvergenceWarning)
def __main__():
	
	classification_dataset_list = [
		'./data/messidor_features.arff',
		'./data/breast-cancer-wisconsin.data',
		'./data/statlog-australian-credit-data.data',
		'./data/statlog-german-credit-data.data',
		'./data/steel-plates-faults.NNA',
		'./data/seismic-bumps.arff',
		'./data/ThoraricSurgery.arff',
		'./data/yeast.data',
		'./data/adult.data',
		'./data/default_of_credit_card_clients.xls',
	]

	regression_dataset_list = [
		'./data_regression/winequality-red.csv',
		'./data_regression/winequality-white.csv',
		'./data_regression/bike_sharing_hour.csv',
		'./data_regression/Concrete_Data.xls',
		'./data_regression/dataset_Facebook.csv',
		'./data_regression/qsar_aquatic_toxicity.csv',
		'./data_regression/sgemm_product.csv',
		'./data_regression/student-por.csv',
		'./data_regression/communities.data',
		'./data_regression/ACT2_competition_training.npz',
		'./data_regression/ACT4_competition_training.npz',
		'./data_regression/parkinson_train_data.txt'
	]

	for dataset in classification_dataset_list:
		print("\n\n******** {} data ***********\n".format(dataset.split('/')[-1]))
		print("*******************************************\n")
	
		knn_classifier = KnnClassifier(dataset)
		model = Model(model_type=knn_classifier)
		model.perform_experiments(dataset)
	
		svm_classifier = SvmClassifier(dataset)
		model = Model(model_type=svm_classifier)
		model.perform_experiments(dataset)
	
		gaussian_nb_classifier = GaussianNbClassifier(dataset)
		model = Model(model_type=gaussian_nb_classifier)
		model.perform_experiments(dataset)
	
		nn_classifier = MlpClassifier(dataset)
		model = Model(model_type=nn_classifier)
		model.perform_experiments(dataset)
	
		lr_classifier = LogisticRegClassifier(dataset)
		model = Model(model_type=lr_classifier)
		model.perform_experiments(dataset)
	
		dt_classifier = DTClassifier(dataset)
		model = Model(model_type=dt_classifier)
		model.perform_experiments(dataset)
		model.model_type.plot_and_save_tree()
	
		rf_classifier = RfClassifier(dataset)
		model = Model(model_type=rf_classifier)
		model.perform_experiments(dataset)
	
		ada_boost_classifier = ABClassifier(dataset)
		model = Model(model_type=ada_boost_classifier)
		model.perform_experiments(dataset)
	
	for dataset in regression_dataset_list:
		print("\n\n******** {} data ***********\n".format(dataset.split('/')[-1]))
		print("*******************************************\n")
	
		linear_regression = LinearReg(dataset)
		model = Model(model_type=linear_regression, is_regression=True)
		model.perform_experiments(dataset)
	
		decision_tree_regressor = DTRegression(dataset)
		model = Model(model_type=decision_tree_regressor,is_regression=True)
		model.perform_experiments(dataset)
	
		svr_regressor = SvrRegression(dataset)
		model = Model(model_type=svr_regressor,is_regression=True)
		model.perform_experiments(dataset)
	
		gaussian_process_regressor = GaussianPrRegressor(dataset)
		model = Model(model_type=gaussian_process_regressor,is_regression=True)
		model.perform_experiments(dataset)
	
		random_forest_regressor = RfRegressor(dataset)
		model = Model(model_type=random_forest_regressor,is_regression=True)
		model.perform_experiments(dataset)
	
		mlp_regressor = MlpRegression(dataset)
		model = Model(model_type=mlp_regressor,is_regression=True)
		model.perform_experiments(dataset)
	
		ab_regressor = ABRegressor(dataset)
		model = Model(model_type=ab_regressor,is_regression=True)
		model.perform_experiments(dataset)

	cifar = CifarModel()
	cifar.load_and_normalize_data() # load cifar data, combine batches data into single numpy array
	cifar.train_and_test_dt() # train and test decision tree
	cifar.train_and_save_cnn() # train and save cnn
	cifar.test_cnn() # test cnn on test dataset
	cifar.activation_maximization_cnn() # create activation map

__main__()