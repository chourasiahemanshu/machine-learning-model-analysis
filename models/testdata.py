from io import StringIO

from scipy.io import arff
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
import xlrd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import TruncatedSVD
from tempfile import TemporaryFile
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor


# with open("../data_regression/ACT2_competition_training.csv") as f:
#     cols = f.readline().rstrip('\n').split(',') # Read the header line and get list of column names
# # Load the actual data, ignoring first column and using second column as targets.
# X = np.loadtxt("../data_regression/ACT2_competition_training.csv", delimiter=',', usecols=range(2, len(cols)), skiprows=1, dtype=np.uint8)
# y = np.loadtxt("../data_regression/ACT2_competition_training.csv", delimiter=',', usecols=[1], skiprows=1)
# np.savez('mat1.npz', X, y)
# "mat.npz".seek(0) # Only needed here to simulate closing & reopening file

# npzfile = np.load("mat1.npz")
# X=npzfile['arr_0']
# y=npzfile['arr_1']
# value=np.mean(np.var(X,axis=1))
# selector = VarianceThreshold(threshold=int(value))
# print(X.shape)
# X=selector.fit_transform(X)
# print(X.shape)


# svd = TruncatedSVD(n_components=100)
# newSvd=svd.fit(X)
# X=newSvd.fit_transform(X)


# data=pd.read_csv("../data_regression/sgemm_product.csv")
# data['Avg_Run'] = data[['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)']].mean(axis=1)
# data = data.drop(['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'], axis=1)
# print(data)
# X = data.iloc[:, 0:14].values
# y = data.iloc[:, 14].values



# df = pd.read_csv("../data_regression/dataset_Facebook.csv", delimiter=';')
# print(df.info())
# print(df)

# data=pd.read_csv("../data_regression/sgemm_product.csv")
# data['Avg_Run'] = data[['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)']].mean(axis=1)
# data = data.drop(['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'], axis=1)
# print(data)
# X = data.iloc[:, 0:14].values
# y = data.iloc[:, 14].values



# data=pd.read_csv("../data_regression/parkinson_train_data.txt",delimiter=",")
# print(data.info())
# f=open("../data_regression/parkinson_train_data.txt","r")
# cols = f.readline().rstrip('\n').split(',')
# c = StringIO(f.read())
# X = np.loadtxt("../data_regression/parkinson_train_data.txt", delimiter=',', usecols=range(1, len(cols)-2))
# y = np.loadtxt("../data_regression/parkinson_train_data.txt", delimiter=',', usecols=[len(cols)-2])

# df = pd.read_csv("../data_regression/dataset_Facebook.csv", delimiter=';')
# labelencoder = LabelEncoder()
# df["Type"] = labelencoder.fit_transform(df["Type"])
# imp = SimpleImputer(missing_values=np.nan, strategy="mean")
# X = df.iloc[:, 0:7].values
# X = imp.fit_transform(X)
# y = df.iloc[:,10].values
# y = imp.fit_transform(y.reshape(-1, 1))
# y = y.flatten()

# X_train=X[:940,:]
# y_train=y[:940]
# X_test=X[941:,:]
# y_test=y[941:]
# X_train=np.array(data.iloc[0:940,0:27])
# y_train=np.array(data.iloc[0:940,27])
# X_test=np.array(data.iloc[941:1040,0:27])
# y_test=np.array(data.iloc[941:1040,27])
#
#
#

# data = pd.read_csv('../data_regression/winequality-red.csv', delimiter=';')
# data = pd.DataFrame(data)
# list_red=['density', 'citric acid', 'quality']
# # include_red=[fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol"]
# # list_white=['chlorides', 'fixed acidity', 'quality']
# include=['volatile acidity',"citric acid","residual sugar","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
# y = data['quality'].values
# data=data.drop(list_red, axis =1)
# data.head()
# X = data.values
# print(X)
# print(y)
# # print(np.var(X,axis=1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Initialize linear regression model
lModel = LinearRegression(normalize=True, fit_intercept=True, copy_X=True)
lModel.fit(X_train, y_train)
y_output=lModel.predict(X_test)
print(lModel.score(X_test, y_test))


# rides = pd.read_csv('../data_regression/bike_sharing_hour.csv')
# # labelencoder = LabelEncoder()
# # rides['dteday']=labelencoder.fit_transform(rides["dteday"])
# # X = rides.iloc[:, 1:14].values
# # y = rides.iloc[:, 16].values
#
# dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
# for each in dummy_fields:
# 	dummies = pd.get_dummies(rides[each], prefix=each, drop_first=True)
# 	rides = pd.concat([rides, dummies], axis=1)
# fields_to_drop = ['instant', 'dteday', 'season', 'atemp', 'yr', 'registered', 'casual', 'season', 'weathersit',
# 						  'mnth', 'hr', 'weekday']  # remove original features
# data = rides.drop(fields_to_drop, axis=1)
# data = pd.DataFrame(data)
# X = data.iloc[:, 0:51].values
# y = data.iloc[:, 51].values
#
# dtm = DecisionTreeRegressor(max_depth=4,
#                            min_samples_split=5,
#                            max_leaf_nodes=10)



# normalize=True, fit_intercept=True, copy_X=True
#
# # Train the model
# dtm.fit(X_train, y_train)
# y_output=dtm.predict(X_test)
# print(dtm.score(X_test, y_test))
#
# lModel.fit(X_train, y_train)
# y_output=lModel.predict(X_test)
# print(lModel.score(X_test, y_test))
#
# param_grid = {
#               "min_samples_split": [10, 20, 40],
#               "max_depth": [2, 6, 8],
#               "min_samples_leaf": [20, 40, 100],
#               "max_leaf_nodes": [5, 20, 100],
#               }
# grid_cv_dtm = GridSearchCV(dtm, param_grid, cv=5)
#
# grid_cv_dtm.fit(X,y)
# print(grid_cv_dtm.best_score_)


# avx=np.mean(y_output)
# avy=np.mean(y_train)
# num=sum( (y_output-avx)*(y_train-avy) )
# num=num*num
# denom=sum((y_output-avx)*(y_output-avx)) * sum( (y_train-avy)*(y_train-avy) )
# print(num/denom)





# df = pd.read_csv("../data_regression/dataset_Facebook.csv", delimiter=';')
# print(df.info())
# print(df)

# data=pd.read_csv("../data_regression/sgemm_product.csv")
# data['Avg_Run'] = data[['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)']].mean(axis=1)
# data = data.drop(['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'], axis=1)
# print(data)
# X = data.iloc[:, 0:14].values
# y = data.iloc[:, 14].values


# data = pd.read_csv("../data_regression/communities.data",  delimiter=",", header=None)
# print(data.iloc[:, 5:127])
# imp = SimpleImputer(missing_values="?", strategy="most_frequent")
# data=imp.fit_transform(data.iloc[:, 5:127]).astype(float)
# X= data[:,0:121]
# y=data[:,121]






# labelencoder = LabelEncoder()
# df["Type"] = labelencoder.fit_transform(df["Type"])
# imp = SimpleImputer(missing_values=np.nan, strategy="mean")
# X=df.iloc[:,0:16].values
# X= imp.fit_transform(X)
# print(X)
# y=df.iloc[:, 18].values
# print(y)
# y= imp.fit_transform(y.reshape(-1,1))
# y=y.flatten()
# print(y)









# Read student data
# parkinson_data = pd.read_csv("../data_regression/parkinson_train_data.txt")
# print(parkinson_data)
# #Data Exploration
#
# #Number of patients
# n_patients = parkinson_data.shape[0]
# print(n_patients)
#
# #Number of features
# n_features = parkinson_data.shape[1]-1
# print(n_features)
#
# #With Parkinsons
# n_parkinsons = parkinson_data[parkinson_data['status'] == 1].shape[0]
#
# #Without Parkinsons
# n_healthy = parkinson_data[parkinson_data['status'] == 0].shape[0]
#
# #Result Output
# print "Total number of patients: {}".format(n_patients)
# print "Number of features: {}".format(n_features)
# print "Number of patients with Parkinsons: {}".format(n_parkinsons)
# print "Number of patients without Parkinsons: {}".format(n_healthy)
#
# #Preparing the Data
#
# # Extract feature columns
# feature_cols = list(parkinson_data.columns[1:16]) + list(parkinson_data.columns[18:])
# target_col = parkinson_data.columns[17]
#
# # Show the list of columns
# print "Feature columns:\n{}".format(feature_cols)
# print "\nTarget column: {}".format(target_col)
#
# # Separate the data into feature data and target data (X_all and y_all, respectively)
# X_all = parkinson_data[feature_cols]
# y_all = parkinson_data[target_col]
#
# # Show the feature information by printing the first five rows
# print "\nFeature values:"
# print X_all.head()
#
# # Training and Testing Data Split
# num_all = parkinson_data.shape[0]
# num_train = 150 # about 75% of the data
# num_test = num_all - num_train
#
# # Select features and corresponding labels for training/test sets
#
# X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test,random_state=5)
# print "Shuffling of data into test and training sets complete!"
#
# print "Training set: {} samples".format(X_train.shape[0])
# print "Test set: {} samples".format(X_test.shape[0])
#
# X_train_50 = X_train[:50]
# y_train_50 = y_train[:50]
#
# X_train_100 = X_train[:100]
# y_train_100 = y_train[:100]
#
# X_train_150 = X_train[:150]
# y_train_150 = y_train[:150]





# names=[i for i in range(0,128)]
# data = pd.read_csv('../data_regression/communities.data',names=names, delimiter=",")
# print(data.head(20))
# for i in range(0,128):
#     print(data[i].dtype)
#
# imp = SimpleImputer()
# dataset = pd.DataFrame(data)
# dataset = imp.fit_transform(dataset)
# print(dataset.info())
# print(dataset.describe())






# def Weekday(x):
#     if x == 1:
#         return 'Su'
#     elif x== 2:
#         return 'Mo'
#     elif x == 3:
#         return 'Tu'
#     elif x == 4:
#         return 'We'
#     elif x == 5:
#         return 'Th'
#     elif x ==6:
#         return 'Fr'
#     elif x == 7:
#         return "Sa"




# df['Weekday'] = df['Post Weekday'].apply(lambda x: Weekday(x))
# dayDf = pd.get_dummies(df['Weekday'])
# df = pd.concat([df,dayDf],axis=1)
# hours = list(range(0,18))
# #hours
# for i in hours:
#     hours[i] = str(hours[i])
#     hours[i]='hr_'+ hours[i]
# hourDf = pd.get_dummies(df['Post Hour'],prefix='hr_')
# df = pd.concat([df,hourDf],axis=1)
# monthDf = pd.get_dummies(df['Post Month'],prefix='Mo')
# df = pd.concat([df,monthDf],axis=1)
# df['Video'] = pd.get_dummies(df['Type'])['Video']
# df['Status'] = pd.get_dummies(df['Type'])['Status']
# df['Photo'] = pd.get_dummies(df['Type'])['Photo']
# df['Cat_1'] = pd.get_dummies(df['Category'])[1]
# df['Cat_2'] = pd.get_dummies(df['Category'])[2]
# imp = SimpleImputer(missing_values=np.nan, strategy="mean")
# # df=pd(imp.fit_transform(df))
# df=df.fillna(0)
# X = df[['Page total likes','Paid','Video','Status','Photo',
#     'Cat_1','Cat_2','Mo','Tu','Sa',"We",'Th','Fr',
#        'hr__17','hr__1','hr__2','hr__3','hr__4','hr__5', 'hr__6','hr__7','hr__8',
#         'hr__9','hr__10','hr__11','hr__12','hr__13','hr__14','hr__15','hr__16','Mo_1',
#        'Mo_2','Mo_12','Mo_4','Mo_5','Mo_6','Mo_7','Mo_8','Mo_9','Mo_11','Mo_10']]
# y = df['like']
#
# print(X)
# print(y)
















# filepath= '../data_regression/Concrete_Data.xls'
# input_data = pd.read_excel(filepath)
# X = input_data.iloc[:, 0:8].values
# y = input_data.iloc[:, 8].values




# data_path = 'Bike-Sharing-Dataset/hour.csv'
# rides = pd.read_csv('../data_regression/bike_sharing_hour.csv')
# print(rides.head(2))
# dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
# for each in dummy_fields:
#     dummies = pd.get_dummies(rides[each], prefix=each, drop_first=True)
#     rides = pd.concat([rides, dummies], axis=1)
# print(rides.head(2))
# fields_to_drop = ['instant', 'dteday', 'season', 'atemp', 'yr', 'registered', 'casual', 'season',
#                   'weathersit', 'mnth', 'hr', 'weekday'] #remove original features
# data = rides.drop(fields_to_drop, axis=1)
# print(data.head(2))
# data = pd.DataFrame(data)
# X = data.iloc[:, 0:51].values
# y = data.iloc[:, 51].values
#




# data = pd.read_excel('../data/default_of_credit_card_clients.xls')
# # print(data.head(20))
# dataset=pd.DataFrame(data)
# dataset=dataset.iloc[1:,1:]
# print(dataset)
# print(dataset.iloc[:,0:23].values.astype(int))
# print(dataset.iloc[:,23].values).astype(int)




# data = arff.loadarff('../data/messidor_features.arff')
# dataset = pd.DataFrame(data[0])
# print(dataset)
# X = dataset.iloc[:, 0: 19].values
# y = dataset.iloc[:, 19].values.astype('int') # convert to int from object type
# print(X)

# data = arff.loadarff('../data/seismic-bumps.arff')
# dataset = pd.DataFrame(data[0])
# label_encoder = LabelEncoder()
# dataset['seismic'] = label_encoder.fit_transform(dataset['seismic'])
# dataset['seismoacoustic'] = label_encoder.fit_transform(dataset['seismoacoustic'])
# dataset['shift'] = label_encoder.fit_transform(dataset['shift'])
# dataset['ghazard'] = label_encoder.fit_transform(dataset['ghazard'])
# dataset['class'] = label_encoder.fit_transform(dataset['class'])
# print(dataset)
# X = dataset.iloc[:, 0: 18].values.astype(int)
# y = dataset.iloc[:, 18].values
# print(X)
# print(y)


# data = arff.loadarff('../data/ThoraricSurgery.arff')
# dataset = pd.DataFrame(data[0])
# label_encoder = LabelEncoder()
# dataset['DGN'] = label_encoder.fit_transform(dataset['DGN'])
# dataset['PRE6'] = label_encoder.fit_transform(dataset['PRE6'])
# dataset['PRE7'] = label_encoder.fit_transform(dataset['PRE7'])
# dataset['PRE8'] = label_encoder.fit_transform(dataset['PRE8'])
# dataset['PRE9'] = label_encoder.fit_transform(dataset['PRE9'])
# dataset['PRE10'] = label_encoder.fit_transform(dataset['PRE10'])
# dataset['PRE11'] = label_encoder.fit_transform(dataset['PRE11'])
# dataset['PRE14'] = label_encoder.fit_transform(dataset['PRE14'])
# dataset['PRE17'] = label_encoder.fit_transform(dataset['PRE17'])
# dataset['PRE19'] = label_encoder.fit_transform(dataset['PRE19'])
# dataset['PRE25'] = label_encoder.fit_transform(dataset['PRE25'])
# dataset['PRE30'] = label_encoder.fit_transform(dataset['PRE30'])
# dataset['PRE32'] = label_encoder.fit_transform(dataset['PRE32'])
# dataset['Risk1Yr']= label_encoder.fit_transform(dataset['Risk1Yr'])
# print(dataset)
# X = dataset.iloc[:, 0: 16].values
# y = dataset.iloc[:, 16].values
# print(X)
# print(y)


# names = ['Sequence Name','mcg', 'gvh', 'alm', 'mit', 'erl','pox','vac','nuc','class']
# data = pd.read_csv('../data/yeast.data',names=names, delim_whitespace=True)
# # print(data.head(20))
# dataset=pd.DataFrame(data)
# print(dataset)
# label_encoder = LabelEncoder()
# dataset['Sequence Name']=label_encoder.fit_transform(dataset['Sequence Name'])
# dataset['class']=label_encoder.fit_transform(dataset['class'])
# print(dataset.iloc[:,0:9].values.astype(int))
# print(dataset.iloc[:,9].values)







