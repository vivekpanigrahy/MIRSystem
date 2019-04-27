import numpy as np
import helpers.FileReader
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import sklearn.svm
from sklearn.model_selection import train_test_split as tts
import multiprocessing

from imblearn.over_sampling import SMOTE
from collections import Counter

target_names = None

use_PCA = True
use_SMOTE = True
smote_threshold = 3000


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
    X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2, random_state=42)

    c = [0.01, 0.1, 1, 10, 100, 1000]
    gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    kernel = ["rbf", "poly"]
    for k in range(len(kernel)):
        accuracies = []
        c_dict = {}
        gamma_dict = {}
        model_dict = {}
        for i in range(len(c)):
            for j in range(len(gamma)):
                clf = sklearn.svm.SVC(C=c[i], gamma=gamma[j], kernel=kernel[k])
                print("model made")
                clf.fit(X_train, Y_train)
                print("model fit")
                YPred_val = clf.predict(X_test)
                print("prediction done")
                acc = accuracy_score(Y_test, YPred_val)
                accuracies.append(acc * 100)
                c_dict.update({acc * 100: c[i]})
                gamma_dict.update({acc * 100: gamma[j]})
                model_dict.update({acc * 100: kernel[k]})
                print("Test accuracy Using " + kernel[k] + " for C = " + str(c[i]) + " and Gamma = " + str(
                    gamma[j]) + " is " + str(acc * 100))
        print(c_dict)
        print(gamma_dict)




if __name__ == '__main__':
    X, Y = preprocessing()
    print("after preprocessing")
    print("X shape", X.shape)
    print(X)
    print("Y shape", Y.shape)
    print(Y)
    main(X, Y)