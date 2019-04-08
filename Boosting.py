import numpy as np
import helpers.FileReader
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import time
from sklearn.linear_model import LogisticRegression
import multiprocessing

from imblearn.over_sampling import SMOTE
from collections import Counter

# 40432 samples

target_names = None


def preprocessing():
    global target_names

    data = helpers.FileReader.loadCsv('data/dataSet.csv')
    labels = helpers.FileReader.loadCsv('data/labels.csv')
    labels.rename(columns={0: 'label'}, inplace=True)
    target_names = list(labels.label.value_counts().index)
    #print("Unique labels: ", target_names)
    print(labels.label.value_counts())

    X = data.values  # [0:1000, :]
    Y = labels.values.ravel()  # [0:1000]

    # --- Dim. Reduction ---
    # print("Num initial features: ", X.shape[1])
    # pca = PCA(n_components=0.975, svd_solver='full')
    # pca.fit(X)
    # print(pca.explained_variance_ratio_)
    # X = pca.transform(X)
    # print("Num latent features: ", X.shape[1])

    # --- SMOTE ---
    # Create dict, where for each class we specify # of samples we desire
    smote_targets = {k: v for k, v in zip(target_names, labels.label.value_counts().values)}
    # Make each class have at least 5000 samples
    for k, v in smote_targets.items():
        if v < 5000:
            smote_targets[k] = 5000
    # print("smote targets: ", smote_targets)

    print('Original dataset shape %s' % Counter(Y))
    sm = SMOTE(sampling_strategy=smote_targets, random_state=42, n_jobs=multiprocessing.cpu_count() - 1)
    X_res, Y_res = sm.fit_resample(X, Y)
    print('Resampled dataset shape %s' % Counter(Y_res))

    return X_res, Y_res


def main(X, Y):

    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                               n_estimators=50,
                               learning_rate=1)

    # model = LogisticRegression(n_jobs=8)


    kf = KFold(n_splits=10, shuffle=True)

    print("Starting KFold...")
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        pred = model.fit(X_train, Y_train).predict(X_test)
        acc = accuracy_score(y_true=Y_test, y_pred=pred)
        print("------ Fold report ------")
        print("\t# training: ", X_train.shape[0])
        print("\t# test: ", X_test.shape[0])
        print("\tAcc: ", acc)
        print(classification_report(Y_test, pred, target_names), "\n\n")
        exit(0)


if __name__ == '__main__':
    X, Y = preprocessing()
    start_time = time.clock()
    main(X, Y)
    print("Seconds elapsed - ", str(time.clock() - start_time))