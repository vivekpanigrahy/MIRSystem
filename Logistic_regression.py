from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import csv
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
from sklearn.model_selection import train_test_split, KFold
import utils
import multiprocessing
from collections import Counter
from imblearn.over_sampling import SMOTE

smote_threshold = 3000
target_names = None

#Load Data
def load_data():
    data = loadCsv('data/dataSetMedium.csv')
    labels = loadCsv('data/labelsMedium.csv')
    
    Y = labels.values[:, 1].ravel()
    X = data.values
    X = X[4:, 1:]
    
    print(X.shape[0])
    print(X.shape[1])
    
    return X, Y
    
def loadCsv(fileName):
    dataset = pd.read_csv(fileName, sep=",", header=None)
    return dataset

def load_smote_data():
	data = loadCsv('data/dataSetMedium.csv')
	labels = loadCsv('data/labelsMedium.csv')

	labels.rename(columns={1: 'label'}, inplace=True)
	target_names = list(labels.label.value_counts().index)

	Y = labels.values[:, 1].ravel()
	X = data.values
	X = X[4:, 1:]

	smote_targets = {k: v for k, v in zip(target_names, labels.label.value_counts().values)}
	for k, v in smote_targets.items():
		if v < smote_threshold:
			smote_targets[k] = smote_threshold
			print("smote targets: ", smote_targets)

	print('Original dataset shape %s' % Counter(Y))
	sm = SMOTE(sampling_strategy=smote_targets, k_neighbors=5, random_state=42, n_jobs=multiprocessing.cpu_count() - 1)

	X_res, Y_res = sm.fit_resample(X, Y)
	print('Resampled dataset shape %s' % Counter(Y_res))

	return X_res, Y_res


def main():
    #X, Y = load_data()
    X, Y = load_smote_data()
    
    kf = KFold(n_splits=10, shuffle=True)
    
    logisticRegr = LogisticRegression(solver='sag', n_jobs=3)
    
    predictions = []
    scores= []
    count = 0
    
    output_file = open("output_lr.txt", "w+")

    for train_idx, test_idx in kf.split(X):
        print("Running For Count: "+ str(count))
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        print(X_train.shape)
        print(X_test.shape)
        print(Y_train.shape)

        logisticRegr.fit(X_train, Y_train)
        output_file.write(str(logisticRegr.score(X_test, Y_test)))
        output_file.write("\n")
        count = count+1
        
   
    #output_file.write(predictions)
    output_file.close()
    

main()