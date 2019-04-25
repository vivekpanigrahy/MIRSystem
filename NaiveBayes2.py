import multiprocessing
from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

import helpers.FileReader

target_names = None
use_SMOTE = True
smote_threshold = 1000


def preprocessing():
    global target_names
    data = helpers.FileReader.loadCsv('data/dataSetMedium.csv')
    labels = helpers.FileReader.loadCsv('data/labelsMedium.csv')

    labels.rename(columns={1: 'label'}, inplace=True)
    target_names = list(labels.label.value_counts().index)
    print("Unique labels: ", target_names)
    print(labels.label.value_counts())

    X = data.values
    X = X[4:, 1:]
    Y = labels.values[:, 1].ravel()
    print("Number of unique classes: ", len(np.unique(Y)))
    print("Shape of data: ", X.shape)

    X_res = X
    Y_res = Y

    if use_SMOTE:
        smoteTargets = {k: v for k, v in zip(target_names, labels.label.value_counts().values)}
        for k, v in smoteTargets.items():
            if v < smote_threshold:
                smoteTargets[k] = smote_threshold
        print("smote targets: ", smoteTargets)

        print('Original dataset shape %s' % Counter(Y))
        sm = SMOTE(sampling_strategy=smoteTargets, k_neighbors=5, random_state=42,
                   n_jobs=multiprocessing.cpu_count() - 1)

        X_res, Y_res = sm.fit_resample(X, Y)
        print('Resampled dataset shape %s' % Counter(Y_res))

    return X_res, Y_res


def main(X, Y, outputPath):
    kf = KFold(n_splits=5, shuffle=True)
    f = open(outputPath, 'w')

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


if __name__ == '__main__':
    X, Y = preprocessing()
    outputPath = 'results/output.txt'
    main(X, Y, outputPath)
