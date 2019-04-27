import multiprocessing

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

import helpers.FileReader
import helpers.dataFractions

target_names = None

use_PCA = False
use_SMOTE = False
smote_threshold = 1000


def preprocessing():
    global target_names
    data = helpers.FileReader.loadCsv('data/dataSetMedium.csv')
    labels = helpers.FileReader.loadCsv('data/labelsMedium.csv')
    labels.rename(columns={1: 'label'}, inplace=True)
    target_names = list(labels.label.value_counts().index)

    X = data.values
    X = X[4:, 1:]
    Y = labels.values[:, 1].ravel()

    if use_PCA:
        pca = PCA(n_components=0.2, svd_solver='full')
        pca.fit(X)
        X = pca.transform(X)

    X_res = X
    Y_res = Y

    if use_SMOTE:
        smoteTargets = {k: v for k, v in zip(target_names, labels.label.value_counts().values)}
        for k, v in smoteTargets.items():
            if v < smote_threshold:
                smoteTargets[k] = smote_threshold

        sm = SMOTE(sampling_strategy=smoteTargets, k_neighbors=5, random_state=42,
                   n_jobs=multiprocessing.cpu_count() - 1)

        X_res, Y_res = sm.fit_resample(X, Y)

    return X_res, Y_res


def kFold(nSplit, X, Y, outputPath):
    kf = KFold(n_splits=nSplit, shuffle=True)
    f = open(outputPath+"KFold.txt", 'w')

    for train_idx, test_idx in kf.split(X):
        gnb = GaussianNB()
        #mnb = MultinomialNB()

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        pred = gnb.fit(X_train, Y_train).predict(X_test)
        #pred = mnb.fit(X_train, Y_train).predict(X_test)
        acc = accuracy_score(y_true=Y_test, y_pred=pred)
        f.write("------ Fold report ------ \n\t# training: " +
                str(X_train.shape[0]) + "\n \t# test: " + str(X_test.shape[0])
                + "\n \tAcc: " + str(acc) + " \n " + str(classification_report(Y_test, pred, target_names)) + "\n")
        exit(0)
    f.close()

def fullData(X , Y ):
    gnb = GaussianNB()
    y_pred = gnb.fit(X, Y).predict(X)
    f = open(outputPath + "fullData.txt", 'w')
    f.write("Full Data from train and test Error Rate : " + str(((Y != y_pred).sum()) * 100 / X.shape[0]))
    f.close()

def dataByFrac(X,Y, testFraction):
    gnb = GaussianNB()
    total = list(zip(X,Y))
    trainData, testData = helpers.dataFractions.getFraction(total, testFraction)
    y_pred = gnb.fit(np.array(trainData)[:,:-1].ravel(), np.array(trainData)[:,-1]).predict(np.array(testData)[:,:-1].ravel())
    f = open(outputPath + "DataFrac.txt", 'w')
    f.write("Full Data from train and test Error Rate : " + str(((Y != y_pred).sum()) * 100 / X.shape[0]))
    f.close()
    #print("Test data as Fraction from train Error Rate : %d" % (((testData.iloc[:,-1] != y_pred).sum())*100/testData.shape[0]))



if __name__ == '__main__':
    X, Y = preprocessing()
    outputPath = 'results/'
    nSplit = 10
    testFraction = 0.6
    kFold(nSplit, X, Y, outputPath)
    #fullData(X,Y)
    #dataByFrac(X,Y,testFraction)


