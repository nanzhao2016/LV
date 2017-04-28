from features import feature_select_Percentile, feature_select_KBest, feature_select_TruncatedSVD, feature_select_NMF
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import LinearSVC
from time import time
import sys, os 
from scores import score_function, score_accuracy
	
def modeling_percentil(function, percentile):
	t0 = time()
	data_train, data_test, target_train, target_test = feature_select_Percentile(function, percentile)
	clf = LinearSVC()
	parameters = [{'C': [100, 1000, 10000]}]
	grid = GridSearchCV(clf, parameters,cv=5, verbose=10)
	grid.fit(data_train, target_train)
	print(grid.best_params_)
	print(grid.best_score_)
	pred = grid.predict(data_test)
	accuracy = score_accuracy(pred, target_test)
	print(accuracy)
	roc = score_function(pred, target_test)
	print(roc)
	print("training time is: ", time()-t0, "s")
	return(accuracy, roc)

def modeling_KBest(function, k):
	t0 = time()
	data_train, data_test, target_train, target_test = feature_select_KBest(function, k)
	clf = LinearSVC()
	parameters = [{'C': [100, 1000, 10000]}]
	grid = GridSearchCV(clf, parameters, cv=5, verbose=10)
	grid.fit(data_train, target_train)
	print(grid.best_params_)
	print(grid.best_score_)
	pred = grid.predict(data_test)
	accuracy = score_accuracy(pred, target_test)
	print(accuracy)
	roc = score_function(pred, target_test)
	print(roc)
	print("training time is: ", time()-t0, "s")
	return(accuracy, roc)

def modeling_TruncatedSVD(n_components):
	t0 = time()
	data_train, data_test, target_train, target_test = feature_select_TruncatedSVD(n_components)
	clf = LinearSVC()
	parameters = [{'C': [100, 1000, 10000]}]
	grid = GridSearchCV(clf, parameters, cv=5, verbose=10)
	grid.fit(data_train, target_train)
	print(grid.best_params_)
	print(grid.best_score_)
	pred = grid.predict(data_test)
	accuracy = score_accuracy(pred, target_test)
	print(accuracy)
	roc = score_function(pred, target_test)
	print(roc)
	print("training time is: ", time()-t0, "s")
	return(accuracy, roc)