from scipy.io import arff
import os
import random


try:
	import pandas as pd
except ImportError:
	print("installing pandas---->")
	os.system("conda install pandas")
	print("installation complete---->")
	import pandas as pd


try:
	import xlrd
except ImportError:
	print("installing xlrd---->")
	os.system("conda install xlrd	")
	print("installation complete---->")

try:
	import seaborn as sns
except ImportError:
	print("installing seaborn---->")
	os.system("conda install seaborn")
	import seaborn as sns

# try:
#     import graphviz
# except:
#     print ("Graphiz not found, Installing Grpahiz ")
#     os.system("conda install -c anaconda graphviz")
#     import graphviz

# try:
#     import pydotplus
# except:
#     print("pydotplus not found, Installing pydotplus ")
#     os.system("conda install -c  conda-forge pydotplus")
#     import pydotplus


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, average_precision_score, precision_recall_curve

TEST_SIZE_SPLIT = 0.3
CV_FOLD = 5
CV_FOLD_FOR_LARGE_DATASET = 2

class Model():

	def __init__(self, model_type, is_regression=False):
		self.model_type = model_type
		self.best_classifier = None
		self.best_score = None
		self.is_normalized = False
		self.cv_fold = CV_FOLD
		self.is_regression = is_regression

	def get_classification_data(self, file_path):
		if file_path.endswith('arff'):
			if file_path.endswith('messidor_features.arff'):
				self.get_arff_data(file_path)
			elif file_path.endswith('seismic-bumps.arff'):
				self.get_seismic_bumps_arff_data(file_path)
			else:
				self.get_ThoraricSurgery_arff_data(file_path)
		elif file_path.endswith('data'):
			if file_path.endswith('adult.data'):
				self.get_adult_data(file_path)
			elif file_path.endswith('wisconsin.data'):
				self.get_bc_wisconsin_data(file_path)
			elif 'german' in file_path:
				self.get_german_credit_data(file_path)
			elif 'australian' in file_path:
				self.get_au_credit_data(file_path)
			else:
				self.get_yeast_data(file_path)
		elif file_path.endswith('NNA'):
			self.get_steel_plates_faults_data(file_path)
		elif file_path.endswith('xls'):
			self.get_default_of_credit_cards_clients_data(file_path)
		else:
			print("Don't know how to load this data")

	def get_regression_data(self, file_path):
		if file_path.endswith('csv'):
			if 'winequality' in file_path:
				self.get_wine_quality_data(file_path)
			elif file_path.endswith('hour.csv'):
				self.get_bike_sharing_data(file_path)
			elif file_path.endswith('Facebook.csv'):
				self.get_facebook_metrics_data(file_path)
			elif file_path.endswith('toxicity.csv'):
				self.get_qsar_aquatic_toxicity_data(file_path)
			elif file_path.endswith('student-por.csv'):
				self.get_studentpor_data(file_path)
			else:
				self.get_sgemm_product(file_path)
		elif file_path.endswith('xls'):
			self.get_concrete_data(file_path)
		elif file_path.endswith('data'):
			self.get_communities_and_crime_data(file_path)
		elif file_path.endswith('npz'):
			self.get_merck_molecule_activity(file_path)
		elif file_path.endswith('txt'):
			self.get_parkinson_speech(file_path)
		else:
			print("Don't know how to load this data")


	def get_merck_molecule_activity(self,file_path):
		npzfile = np.load(file_path)
		X = npzfile['arr_0']
		self.y = npzfile['arr_1']
		value = np.mean(np.var(X, axis=1))
		selector = VarianceThreshold(threshold=int(value))
		self.X = selector.fit_transform(X)

	def get_parkinson_speech(self,file_path):
		data = pd.read_csv(file_path,delimiter=",")
		self.X = np.array(data.iloc[:, 1:27])
		self.y = np.array(data.iloc[:, 27])


	def get_communities_and_crime_data(self, file_path):
		data = pd.read_csv(file_path, delimiter=",", header=None)
		imp = SimpleImputer(missing_values="?", strategy="most_frequent")
		data = imp.fit_transform(data.iloc[:, 5:127]).astype(float)
		self.X = data[:, 0:121]
		self.y = data[:, 121]

	def get_wine_quality_data(self, file_path):
		data = pd.read_csv(file_path, delimiter=';')
		data = pd.DataFrame(data)
		self.X = data.iloc[:, 0:11].values
		self.y = data.iloc[:, 11].values

	def get_bike_sharing_data(self, file_path):
		rides = pd.read_csv(file_path)
		dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
		for each in dummy_fields:
			dummies = pd.get_dummies(rides[each], prefix=each, drop_first=True)
			rides = pd.concat([rides, dummies], axis=1)
		fields_to_drop = ['instant', 'dteday', 'season', 'atemp', 'yr', 'registered', 'casual', 'season', 'weathersit',
						  'mnth', 'hr', 'weekday']  # remove original features
		data = rides.drop(fields_to_drop, axis=1)
		data = pd.DataFrame(data)
		self.X = data.iloc[:, 0:51].values
		self.y = data.iloc[:, 51].values

	def get_concrete_data(self, file_path):
		input_data = pd.read_excel(file_path)
		self.X = input_data.iloc[:, 0:8].values
		self.y = input_data.iloc[:, 8].values

	def get_studentpor_data(self,file_path):
		student_data = pd.read_csv(file_path,delimiter=';')
		data = pd.DataFrame(student_data)
		label_encoder = LabelEncoder();
		# encode label student
		data["school"] = label_encoder.fit_transform(data["school"])
		# sex
		data.iloc[:, 1] = label_encoder.fit_transform(data.iloc[:, 1])
		# address
		data.iloc[:, 3] = label_encoder.fit_transform(data.iloc[:, 3])
		# famsize
		data.iloc[:, 4] = label_encoder.fit_transform(data.iloc[:, 4])
		# Pstatus
		data.iloc[:, 5] = label_encoder.fit_transform(data.iloc[:, 5])
		# Mjob
		data.iloc[:, 8] = label_encoder.fit_transform(data.iloc[:, 8])
		# Fjob
		data.iloc[:, 9] = label_encoder.fit_transform(data.iloc[:, 9])
		# reason
		data.iloc[:, 10] = label_encoder.fit_transform(data.iloc[:, 10])
		# guardian
		data.iloc[:, 11] = label_encoder.fit_transform(data.iloc[:, 11])
		# schoolsup
		data.iloc[:, 15] = label_encoder.fit_transform(data.iloc[:, 15])
		# famsup
		data.iloc[:, 16] = label_encoder.fit_transform(data.iloc[:, 16])
		# paid
		data.iloc[:, 17] = label_encoder.fit_transform(data.iloc[:, 17])
		# activities
		data.iloc[:, 18] = label_encoder.fit_transform(data.iloc[:, 18])
		# nursery
		data.iloc[:, 19] = label_encoder.fit_transform(data.iloc[:, 19])
		# higher
		data.iloc[:, 20] = label_encoder.fit_transform(data.iloc[:, 20])
		# internet
		data.iloc[:, 21] = label_encoder.fit_transform(data.iloc[:, 21])
		# romantic
		data.iloc[:, 22] = label_encoder.fit_transform(data.iloc[:, 22])
		X = data.iloc[:, 0:32]
		fields = ["Fedu", "age", "Fjob", "activities", "famsize", "health", "Walc", "romantic",
				  "goout", "famrel", "Pstatus", "famsup", "nursery", "studytime", "absences", "Mjob", "G2"]
		self.X = X[fields].values
		self.y = data.iloc[:, 32].values

	def Weekday(self,x):
		if x == 1:
			return 'Su'
		elif x == 2:
			return 'Mo'
		elif x == 3:
			return 'Tu'
		elif x == 4:
			return 'We'
		elif x == 5:
			return 'Th'
		elif x == 6:
			return 'Fr'
		elif x == 7:
			return "Sa"


	def get_facebook_metrics_data(self, file_path):
		df = pd.read_csv(file_path, delimiter=';')
		labelencoder = LabelEncoder()
		df["Type"] = labelencoder.fit_transform(df["Type"])
		imp = SimpleImputer(missing_values=np.nan, strategy="mean")
		X = df.iloc[:, 0:7].values
		self.X = imp.fit_transform(X)
		y = df.iloc[:, 10].values
		y = imp.fit_transform(y.reshape(-1, 1))
		self.y = y.flatten()


		# df = pd.read_csv(file_path, delimiter=';')
		# df['Weekday'] = df['Post Weekday'].apply(lambda x: self.Weekday(x))
		# dayDf = pd.get_dummies(df['Weekday'])
		# df = pd.concat([df, dayDf], axis=1)
		# hours = list(range(0, 18))
		# # hours
		# for i in hours:
		# 	hours[i] = str(hours[i])
		# 	hours[i] = 'hr_' + hours[i]
		# hourDf = pd.get_dummies(df['Post Hour'], prefix='hr_')
		# df = pd.concat([df, hourDf], axis=1)
		# monthDf = pd.get_dummies(df['Post Month'], prefix='Mo')
		# df = pd.concat([df, monthDf], axis=1)
		# df['Video'] = pd.get_dummies(df['Type'])['Video']
		# df['Status'] = pd.get_dummies(df['Type'])['Status']
		# df['Photo'] = pd.get_dummies(df['Type'])['Photo']
		# df['Cat_1'] = pd.get_dummies(df['Category'])[1]
		# df['Cat_2'] = pd.get_dummies(df['Category'])[2]
		# df = df.fillna(0)
		# self.X = df[['Page total likes', 'Paid', 'Video', 'Status', 'Photo',
		# 			 'Cat_1', 'Cat_2', 'Mo', 'Tu', 'Sa', "We", 'Th', 'Fr',
		# 			 'hr__17', 'hr__1', 'hr__2', 'hr__3', 'hr__4', 'hr__5', 'hr__6', 'hr__7', 'hr__8',
		# 			 'hr__9', 'hr__10', 'hr__11', 'hr__12', 'hr__13', 'hr__14', 'hr__15', 'hr__16', 'Mo_1',
		# 			 'Mo_2', 'Mo_12', 'Mo_4', 'Mo_5', 'Mo_6', 'Mo_7', 'Mo_8', 'Mo_9', 'Mo_11', 'Mo_10']]
		# self.y = df['like']

	def get_qsar_aquatic_toxicity_data(self, file_path):
		data = pd.read_csv(file_path, delimiter=";")
		self.X = data.iloc[:, 0:8].values
		self.y = data.iloc[:, 8].values

	def get_sgemm_product(self, file_path):
		data = pd.read_csv(file_path)
		data['Avg_Run'] = data[['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)']].mean(axis=1)
		data = data.drop(['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'], axis=1)
		self.X = data.iloc[:, 0:14].values
		self.y = data.iloc[:, 14].values

	def get_adult_data(self, file_path):
		'''
		Replace <=50K as 0 and >50K as 1 to make classification work properly.
		Also, replace string features into numbers
		'''
		data = pd.read_csv(file_path, header=None)
		data = self.change_text_to_number_data_for_adult(data)
		data.replace(' <=50K', 0, inplace=True)
		data.replace(' >50K', 1, inplace=True)
		self.y = np.array(data.iloc[:,14])
		self.X = np.array(data.iloc[:,:14])


	def change_text_to_number_data_for_adult(self, data):
		'''
		Replace textual data in columns with numbers
		'''
		random.seed(0)

		workclass_dict = {' State-gov': 0, ' Self-emp-not-inc': 1, ' Private': 2, ' Federal-gov': 3, ' Local-gov': 4, ' Self-emp-inc': 5, ' Without-pay': 6, ' Never-worked': 7, ' ?': random.randrange(0,8)}
		education_dict = {' Bachelors': 0, ' HS-grad': 1, ' 11th': 2, ' Masters': 3, ' 9th': 4, ' Some-college': 5, ' Assoc-acdm': 6, ' Assoc-voc': 7, ' 7th-8th': 8, ' Doctorate': 9, ' Prof-school': 10, ' 5th-6th': 11, ' 10th': 12, ' 1st-4th': 13, ' Preschool': 14, ' 12th': 15, ' ?': random.randrange(0,16)}
		marital_status_dict = {' Never-married': 0, ' Married-civ-spouse': 1, ' Divorced': 2, ' Married-spouse-absent': 3, ' Separated': 4, ' Married-AF-spouse': 5, ' Widowed': 6, ' ?': random.randrange(0,7)}
		occupation_dict = {' Adm-clerical': 0, ' Exec-managerial': 1, ' Handlers-cleaners': 2, ' Prof-specialty': 3, ' Other-service': 4, ' Sales': 5, ' Craft-repair': 6, ' Transport-moving': 7, ' Farming-fishing': 8, ' Machine-op-inspct': 9, ' Tech-support': 10, ' Protective-serv': 11, ' Armed-Forces': 12, ' Priv-house-serv': 13, ' ?': random.randrange(0,14)}
		relationship_dict = {' Not-in-family': 0, ' Husband': 1, ' Wife': 2, ' Own-child': 3, ' Unmarried': 4, ' Other-relative': 5, ' ?': random.randrange(0,6)}
		race_dict = {' White': 0, ' Black': 1, ' Asian-Pac-Islander': 2, ' Amer-Indian-Eskimo': 3, ' Other': 4, ' ?': random.randrange(0,5)}
		sex_dict = {' Male': 0, ' Female': 1, ' ?': random.randrange(0,2)}
		country_dict = {' United-States': 0, ' Cuba': 1, ' Jamaica': 2, ' India': 3, ' Mexico': 4, ' South': 5, ' Puerto-Rico': 6, ' Honduras': 7, ' England': 8, ' Canada': 9, ' Germany': 10, ' Iran': 11, ' Philippines': 12, ' Italy': 13, ' Poland': 14, ' Columbia': 15, ' Cambodia': 16, ' Thailand': 17, ' Ecuador': 18, ' Laos': 19, ' Taiwan': 20, ' Haiti': 21, ' Portugal': 22, ' Dominican-Republic': 23, ' El-Salvador': 24, ' France': 25, ' Guatemala': 26, ' China': 27, ' Japan': 28, ' Yugoslavia': 29, ' Peru': 30, ' Outlying-US(Guam-USVI-etc)': 31, ' Scotland': 32, ' Trinadad&Tobago': 33, ' Greece': 34, ' Nicaragua': 35, ' Vietnam': 36, ' Hong': 37, ' Ireland': 38, ' Hungary': 39, ' Holand-Netherlands': 40, ' ?': random.randrange(0,41)}
		
		for key, val in workclass_dict.items():
			data[1].replace(key, val, inplace=True)

		for key, val in education_dict.items():
			data[3].replace(key, val, inplace=True)

		for key, val in marital_status_dict.items():
			data[5].replace(key, val, inplace=True)

		for key, val in occupation_dict.items():
			data[6].replace(key, val, inplace=True)

		for key, val in relationship_dict.items():
			data[7].replace(key, val, inplace=True)

		for key, val in race_dict.items():
			data[8].replace(key, val, inplace=True)

		for key, val in sex_dict.items():
			data[9].replace(key, val, inplace=True)

		for key, val in country_dict.items():
			data[13].replace(key, val, inplace=True)
		
		return data

	def process_and_load_adult_test_data(self):
		data = pd.read_csv('./data/adult.test', header=None)
		data = self.change_text_to_number_data_for_adult(data)
		data.replace(' <=50K.', 0, inplace=True)
		data.replace(' >50K.', 1, inplace=True)
		self.X_train = self.X
		self.y_train = self.y
		self.y_test = np.array(data.iloc[:,14])
		self.X_test = np.array(data.iloc[:,:14])

	def get_steel_plates_faults_data(self, file_path):
		'''
		This method loads the data and converts the available data and assign classes as follows
		0 - Pastry, 1 - Z_Scratch, 2 - K_Scatch, 3 - Stains, 4 - Dirtiness, 5 - Bumps, 6 - Other_Faults
		'''
		data = pd.read_csv(file_path, header=None, delim_whitespace=True)
		y_all = np.array(data.iloc[:, 27:])
		y = np.empty(1941, dtype=np.int)
		y[y_all[:,0] == 1] = 0
		y[y_all[:,1] == 1] = 1
		y[y_all[:,2] == 1] = 2
		y[y_all[:,3] == 1] = 3
		y[y_all[:,4] == 1] = 4
		y[y_all[:,5] == 1] = 5
		y[y_all[:,6] == 1] = 6
		self.y = y
		self.X = np.array(data.iloc[:,:27])

	def get_german_credit_data(self, file_path):
		'''
		This method is used to load the data from a file which has .data extension and seperate out X and y labels
		'''
		data = pd.read_csv(file_path, header=None, delim_whitespace=True)
		y = np.array(data[24])
		self.y = y.astype('int')
		self.X = np.array(data.iloc[:, :24])

	def get_au_credit_data(self, file_path):
		'''
		This method is used to load the data from a file which has .data extension and seperate out X and y labels
		'''
		data = pd.read_csv(file_path, header=None, delim_whitespace=True)
		y = np.array(data[14])
		self.y = y.astype('int')
		self.X = np.array(data.iloc[:, :14])

	def get_bc_wisconsin_data(self, file_path):
		'''
		This method is used to load the data from a file which has .data extension and seperate out X and y labels
		'''
		data = pd.read_csv(file_path, header=None)
		y = np.array(data[1])
		y[y == 'M'] = 1
		y[y == 'B'] = 0
		self.y = y.astype('int')
		self.X = np.array(data.iloc[:, 2:])
	
	def get_arff_data(self, file_path):
		'''
		This method is used to load the arff data and seperate out X and y labels
		'''
		data = arff.loadarff(file_path)
		dataset = pd.DataFrame(data[0])
		self.X = dataset.iloc[:, 0: 19].values
		self.y = dataset.iloc[:, 19].values.astype('int') # convert to int from object type

	def get_yeast_data(self, file_path):
		'''
        This method is used to load the data from a file which has .data extension and seperate out X and y labels
        '''
		names = ['Sequence Name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'class']
		data = pd.read_csv(file_path, names=names, delim_whitespace=True)
		dataset = pd.DataFrame(data)
		label_encoder = LabelEncoder()
		dataset['Sequence Name'] = label_encoder.fit_transform(dataset['Sequence Name'])
		dataset['class'] = label_encoder.fit_transform(dataset['class'])
		# print(dataset['class'])
		self.X = dataset.iloc[:, 0:9].values.astype(int)
		self.y = dataset.iloc[:, 9].values

	def get_seismic_bumps_arff_data(self, file_path):
		'''
        This method is used to load the arff data and seperate out X and y labels
        '''
		data = arff.loadarff(file_path)
		dataset = pd.DataFrame(data[0])
		label_encoder = LabelEncoder()
		dataset['seismic'] = label_encoder.fit_transform(dataset['seismic'])
		dataset['seismoacoustic'] = label_encoder.fit_transform(dataset['seismoacoustic'])
		dataset['shift'] = label_encoder.fit_transform(dataset['shift'])
		dataset['ghazard'] = label_encoder.fit_transform(dataset['ghazard'])
		dataset['class'] = label_encoder.fit_transform(dataset['class'])
		self.X = dataset.iloc[:, 0: 18].values.astype(int)
		self.y = dataset.iloc[:, 18].values

	def get_ThoraricSurgery_arff_data(self, file_path):
		'''
        This method is used to load the arff data and seperate out X and y labels
        '''
		data = arff.loadarff(file_path)
		dataset = pd.DataFrame(data[0])
		label_encoder = LabelEncoder()
		dataset['DGN'] = label_encoder.fit_transform(dataset['DGN'])
		dataset['PRE6'] = label_encoder.fit_transform(dataset['PRE6'])
		dataset['PRE7'] = label_encoder.fit_transform(dataset['PRE7'])
		dataset['PRE8'] = label_encoder.fit_transform(dataset['PRE8'])
		dataset['PRE9'] = label_encoder.fit_transform(dataset['PRE9'])
		dataset['PRE10'] = label_encoder.fit_transform(dataset['PRE10'])
		dataset['PRE11'] = label_encoder.fit_transform(dataset['PRE11'])
		dataset['PRE14'] = label_encoder.fit_transform(dataset['PRE14'])
		dataset['PRE17'] = label_encoder.fit_transform(dataset['PRE17'])
		dataset['PRE19'] = label_encoder.fit_transform(dataset['PRE19'])
		dataset['PRE25'] = label_encoder.fit_transform(dataset['PRE25'])
		dataset['PRE30'] = label_encoder.fit_transform(dataset['PRE30'])
		dataset['PRE32'] = label_encoder.fit_transform(dataset['PRE32'])
		dataset['Risk1Yr'] = label_encoder.fit_transform(dataset['Risk1Yr'])
		self.X = dataset.iloc[:, 0: 16].values
		self.y = dataset.iloc[:, 16].values

	def get_default_of_credit_cards_clients_data(self, file_path):
		'''
        This method is used to load the xls data and separate out X and y labels
        '''
		data = pd.read_excel(file_path)
		dataset = pd.DataFrame(data)
		dataset = dataset.iloc[1:, 1:]
		self.X = dataset.iloc[:, 0:23].values.astype(int)
		self.y = dataset.iloc[:, 23].values.astype(int)

	def get_train_and_test_split(self, test_size=TEST_SIZE_SPLIT, stratify=True):
		'''
		Splits the dataset based on the test_size value provided
		sets X_train, X_test, y_train, y_test values for the instance
		'''
		if stratify:
			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=0, stratify=self.y)
		else:
			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=0)

	def get_score_without_any_processing(self):
		'''
			Delegates the score method to individual model and gets the score
		'''
		score = self.model_type.score(self.X_train, self.X_test, self.y_train, self.y_test)
		if self.best_score is None or (self.best_score is not None and score >= self.best_score):
			self.best_score = score
			self.best_classifier = self.model_type.get_sklearn_model_class()

		print('----- {} score without any preprocessing: {} -----'.format(self.model_type, score))

	def preprocess_data_with_scaler(self):
		'''
		Preprocess data using sklearn StandardScaler to normalize the dataset.
		'''
		scaler = StandardScaler().fit(self.X_train)
		self.X_train_scaled = scaler.transform(self.X_train)
		self.X_test_scaled = scaler.transform(self.X_test)

	def score_after_preprocessing(self):
		self.preprocess_data_with_scaler()
		score = self.model_type.score(self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test)
		if self.best_score is None or (self.best_score is not None and score >= self.best_score):
			self.best_score = score
			self.best_classifier = self.model_type.get_sklearn_model_class()
			self.is_normalized = True
		print('----- {} score after normalizing dataset: {} -----'.format(self.model_type, score))

	def train_and_predict_for_best_params(self, values, is_scaled=False):
		model = self.model_type.create_new_instance(values)
		if is_scaled:
			model.fit(self.X_train_scaled, self.y_train)
			model.predict(self.X_test_scaled)
			score = model.score(self.X_test_scaled, self.y_test)
		else:
			model.fit(self.X_train, self.y_train)
			model.predict(self.X_test)
			score = model.score(self.X_test, self.y_test)

		if self.best_score is None or (self.best_score is not None and score >= self.best_score):
			self.best_score = score
			self.best_classifier = model
			self.is_normalized = is_scaled

		return score

	def grid_search_with_cross_validation(self, use_preprocessing=False, k_fold=CV_FOLD):
		'''
		Tries to find optimal value of paramters for a model by using cross validations and cv grid
		'''
		classifier = self.model_type.create_new_instance(values={})
		classifier_gscv = GridSearchCV(classifier, self.model_type.param_grid(), cv=k_fold)
		if use_preprocessing:
			classifier_gscv.fit(self.X_train_scaled, self.y_train)
			print('----- {} best param values using grid search cv for {}-fold cross validation on normalized dataset: {} -----'.format(self.model_type, k_fold, classifier_gscv.best_params_))
			score = self.train_and_predict_for_best_params(values=classifier_gscv.best_params_, is_scaled=True)
			print('----- {} score using grid search for {}-fold cross validation on normalized test dataset: {} -----'.format(self.model_type, k_fold, score))
		else:
			classifier_gscv.fit(self.X_train, self.y_train)
			print('----- {} best param values using grid search for {}-fold cross validation without any preprocessing: {} -----'.format(self.model_type, k_fold, classifier_gscv.best_params_))
			score = self.train_and_predict_for_best_params(values=classifier_gscv.best_params_)
			print('----- {} score using grid search for {}-fold cross validation on test dataset without any preprocessing: {} -----'.format(self.model_type, k_fold, score))
	
	def random_search_with_cross_validation(self, use_preprocessing=False, k_fold=CV_FOLD):
		'''
		Tries to find optimal value of paramters for a model by using cross validations and random search
		'''
		classifier = self.model_type.create_new_instance(values={})
		classifier_rscv = RandomizedSearchCV(classifier, self.model_type.param_grid(is_random=True), cv=k_fold)
		if use_preprocessing:
			classifier_rscv.fit(self.X_train_scaled, self.y_train)
			print('----- {} best param values using random search cv for {}-fold cross validation on normalized dataset: {} -----'.format(self.model_type, k_fold, classifier_rscv.best_params_))
			score = self.train_and_predict_for_best_params(values=classifier_rscv.best_params_, is_scaled=True)
			print('----- {} score using random search cv for {}-fold cross validation on normalized test dataset: {} -----'.format(self.model_type, k_fold, score))
		else:
			classifier_rscv.fit(self.X_train, self.y_train)
			print('----- {} best param values using random search cv for {}-fold cross validation without any preprocessing: {} -----'.format(self.model_type, k_fold, classifier_rscv.best_params_))
			score = self.train_and_predict_for_best_params(values=classifier_rscv.best_params_)
			print('----- {} score using random search cv for {}-fold cross validation on test dataset without any preprocessing: {} -----'.format(self.model_type, k_fold, score))

	def plot_confusion_matrix(self):
		# Creates a confusion matrix
		if self.is_normalized:
			y_pred = self.best_classifier.predict(self.X_test_scaled)
		else:
			y_pred = self.best_classifier.predict(self.X_test)
		
		cm = confusion_matrix(self.y_test, y_pred) 

		# Transform to df for easier plotting
		cm_df = pd.DataFrame(cm)

		plt.figure(figsize=(6,4))
		ax = sns.heatmap(cm, annot=True, fmt="d")
		bottom, top = ax.get_ylim()
		ax.set_ylim(bottom + 0.5, top - 0.5)
		plt.title('{} Accuracy:{:.2f}'.format(str(self.model_type), self.best_score*100))
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.savefig("./plots/{}-{}.png".format(self.model_type.dataset.split('/')[-1], str(self.model_type)))
		plt.close()

	def plot_roc_curve(self):
		if self.is_normalized:
			y_pred_proba = self.best_classifier.predict_proba(self.X_test_scaled)
		else:
			y_pred_proba = self.best_classifier.predict_proba(self.X_test)

		pos_label = None

		if "german-credit-data" in self.model_type.dataset:
			pos_label = 1

		fpr, tpr, _ = roc_curve(self.y_test,  y_pred_proba[:,1], pos_label=pos_label)
		auc = roc_auc_score(self.y_test, y_pred_proba[:,1])

		plt.plot(tpr,fpr)
		plt.title('{} ROC AUC:{:.2f}'.format(str(self.model_type), auc))
		plt.ylabel('False Positive Rate')
		plt.xlabel('True Positive Rate')
		plt.savefig("./plots/{}-{}-roc.png".format(self.model_type.dataset.split('/')[-1], str(self.model_type)))
		plt.close()


	def plot_pr_curve(self):
		if self.is_normalized:
			y_pred_proba = self.best_classifier.predict_proba(self.X_test_scaled)
		else:
			y_pred_proba = self.best_classifier.predict_proba(self.X_test)

		pos_label = None
		if "german-credit-data" in self.model_type.dataset:
			pos_label = 1

		p, r, _ = precision_recall_curve(self.y_test,  y_pred_proba[:,1], pos_label=pos_label)
		auc = average_precision_score(self.y_test, y_pred_proba[:,1])

		plt.plot(r,p)
		plt.title('{} Average PR AUC:{:.2f}'.format(str(self.model_type), auc))
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.savefig("./plots/{}-{}-pr.png".format(self.model_type.dataset.split('/')[-1], str(self.model_type)))
		plt.close()
	
	def perform_experiments(self, file_path):
		
		if self.is_regression:
			self.get_regression_data(file_path)
			self.get_train_and_test_split(stratify=False)
		else:
			self.get_classification_data(file_path)
			#adult dataset has its own test dataset
			if file_path != "./data/adult.data":
				self.get_train_and_test_split()
			else:
				self.process_and_load_adult_test_data()

		# reduce cv fold for large datasets inorder to minimize run time
		if file_path == './data/default_of_credit_card_clients.xls' or file_path == './data/adult.data':
			self.cv_fold = CV_FOLD_FOR_LARGE_DATASET
		
		self.get_score_without_any_processing()
		self.score_after_preprocessing()
		
		if file_path != "./data/adult.data" and file_path != './data/default_of_credit_card_clients.xls':
			self.grid_search_with_cross_validation(k_fold=self.cv_fold)
			self.grid_search_with_cross_validation(use_preprocessing=True, k_fold=self.cv_fold)
		self.random_search_with_cross_validation(k_fold=self.cv_fold)
		self.random_search_with_cross_validation(use_preprocessing=True, k_fold=self.cv_fold)
		
		if not self.is_regression:
			if np.unique(self.y).shape[0] == 2:
				self.plot_roc_curve()
				self.plot_pr_curve()
			self.plot_confusion_matrix()

	def perform_experiment_for_cifar(self, X_train, X_test, y_train, y_test):
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test

		self.cv_fold = CV_FOLD_FOR_LARGE_DATASET
		print("without preprocessing")
		self.get_score_without_any_processing()
		self.score_after_preprocessing()
		self.random_search_with_cross_validation(k_fold=self.cv_fold)
		self.random_search_with_cross_validation(use_preprocessing=True, k_fold=self.cv_fold)

