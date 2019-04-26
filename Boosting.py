import numpy as np
import helpers.FileReader
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import time
import multiprocessing
import pickle

from imblearn.over_sampling import SMOTE
from collections import Counter

target_names = None


use_PCA = False
use_SMOTE = True
smote_threshold = 3000
output_folder = 'adaboost_outputs'


def preprocessing():
    global target_names

    data = helpers.FileReader.loadCsv('data/dataSetMedium.csv')
    labels = helpers.FileReader.loadCsv('data/labelsMedium.csv')

    labels.rename(columns={1: 'label'}, inplace=True)
    target_names = list(labels.label.value_counts().index)
    print("Unique labels: ", target_names)
    print(labels.label.value_counts())

    # If you want to use trackIDs, don't splice out below
    X = data.values
    X = X[4:, 1:] # remove columns and track IDs
    Y = labels.values[:, 1].ravel() # remove track IDs


    print("Number of unique classes: ", len(np.unique(Y)))
    print("Shape of data: ", X.shape)

    # --- Dim. Reduction ---
    if use_PCA:
        print("Using PCA...\nNum initial features: ", X.shape[1])
        pca = PCA(n_components=0.975, svd_solver='full')
        pca.fit(X)
        print(pca.explained_variance_ratio_)
        X = pca.transform(X)
        print("Num latent features: ", X.shape[1])

    # --- SMOTE ---
    X_res = X
    Y_res = Y

    if use_SMOTE:
        # Create dict, where for each class we specify # of samples we desire
        smote_targets = {k: v for k, v in zip(target_names, labels.label.value_counts().values)}
        # Make each class have at least smote_threshold samples
        for k, v in smote_targets.items():
            if v < smote_threshold:
                smote_targets[k] = smote_threshold
        print("smote targets: ", smote_targets)

        print('Original dataset shape %s' % Counter(Y))
        sm = SMOTE(sampling_strategy=smote_targets, k_neighbors=5, random_state=42,
                   n_jobs=multiprocessing.cpu_count() - 1)

        X_res, Y_res = sm.fit_resample(X, Y)
        print('Resampled dataset shape %s' % Counter(Y_res))

    return X_res, Y_res


def main(X, Y):

    kf = KFold(n_splits=10, shuffle=True)
    num_estimators = [600] # 25, 50, 75, 100, 125, 150]
    acc_scores = []
    count = 0


    for train_idx, test_idx in kf.split(X):
        curr_num = num_estimators[count]
        print("Starting Fold... num_estimators=", curr_num)
        model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                                   n_estimators=curr_num,
                                   learning_rate=1)

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        pred = model.fit(X_train, Y_train).predict(X_test)
        print("Classification error per estimator: ", model.estimator_errors_)
        acc = accuracy_score(y_true=Y_test, y_pred=pred)
        acc_scores.append(acc)
        # print("------ Fold report ------")
        # print("\t# training: ", X_train.shape[0])
        # print("\t# test: ", X_test.shape[0])
        print("\tAcc: ", acc)
        # print(classification_report(Y_test, pred, target_names), "\n\n")
        count += 1
        if count == len(num_estimators):
            with open(output_folder + '/accuraciesTest_' + str(smote_threshold) + '.pickle', 'wb') as handle:
                pickle.dump(acc_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
            exit(0)



if __name__ == '__main__':
    X, Y = preprocessing()
    start_time = time.clock()
    main(X, Y)
    print("Seconds elapsed - ", str(time.clock() - start_time))