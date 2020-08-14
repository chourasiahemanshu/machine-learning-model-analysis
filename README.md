# machine-learning-model-analysis
Machine Learning Project to see the analysis of different types of models of similar data sets


The goal of this project is to give students experience in the following aspects of applied machine learning:
------------------------------------------------------------------------------------------------------------
```
collecting data sets that are given in diverse formats;
preparing data sets for training and testing;
training many types of classification and regression models;
performing hyperparameter search for each type of model;
evaluating and summarizing classification and regression performance;
generating plots that summarize performance;
different approaches to interpretability; and
writing about machine learning concepts, experiments, and results.
```

Classification models
---------------------------------------------------------------------
```
You should evaluate the following classification models:
k-nearest neighbours classification
Support vector classification
Decision tree classification
Random forest classification
AdaBoost classification
Logistic regression (for classification)
Gaussian naive Bayes classification
Neural network classification
Each of these is provided by scikit-learn under a unified interface. For example, MLPClassifier implements a
fully-connected neural network classifier (also called a multi-layer perceptron, or MLP, and GaussianNB
implements a Gaussian naive Bayes classifier. The AdaBoostClassifier implements AdaBoost for classification,
for which using the default base_estimator is OK to use. Even though a model like logistic regression is not
strictly a classifier, the scikit-learn implementation will still predict class labels.

Hyperparameters. Some types of models have more hyperparameters than others. You do not need to try
every hyperparameter. Just choose 13 hyperparameters that are likely to have impact, such as and for
SVM, or max_depth and n_estimators for random forests, or hidden_layer_sizes and learning_rate and max_iter
for neural networks. You need to choose and justify your strategy for picking hyperparameter ranges and for
sampling the hyperparameters during hyperparameter search, and you should specify how you trained your
final model once the best hyperparameters were found.

Classification datasets
You should evaluate each of the above classification model families on each the following UCI repository
datasets:
1. Diabetic Retinopathy
2. Default of credit card clients
3. Breast Cancer Wisconsin
4. Statlog Australian credit approval)
5. Statlog German credit data) (recommend german.doc for instructions and german-numeric for data.)
6. Steel Plates Faults
7. Adult
8. Yeast
9. Thoracic Surgery Data
10. Seismic-Bumps
For these datasets you'll need to read the data set descriptions and discern which fields are intended to be
features and which are the class labels to be predicted. If a dataset does not come with an explicit train/test
split, then you will have to ensure your methodology can still evaluate the performance of the model on heldout
data. Your conclusions regarding classification should draw from training and evaluating on the above
datasets.
```

Regression models
---------------------------------------------------------------------------------------
```You should evaluate the following regression models:

Support vector regression
Decision tree regression
Random forest regression
AdaBoost regression
Gaussian process regression
Linear regression
Neural network regression
Each of these is provided by scikit-learn. For example, the MLPRegressor class implements a fully-connected
neural network for regression. The SVR class implements support vector regression. The AdaBoostRegressor
implements AdaBoost for regression, for which using the default base_estimator is OK. The
GaussianProcessRegressor implements Gaussian process regression.


Regression datasets
You should evaluate each of the above regression model families on each the following datasets, which are
mostly again from the UCI repository:
1. Wine Quality
2. Communities and Crime
3. QSAR aquatic toxicity
4. Parkinson Speech (extract RAR files with 7zip for Windows/Linux or Unarchiver for Mac.)
5. Facebook metrics
6. Bike Sharing (use hour data)
7. Student Performance (use just student-por.csv if you do not know how to merge the math grades)
8. Concrete Compressive Strength
9. SGEMM GPU kernel performance
10. Merck Molecular Activity Challenge (from Kaggle)




```
