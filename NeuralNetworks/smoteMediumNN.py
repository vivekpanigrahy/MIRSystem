import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import sklearn.svm
import time
from sklearn.linear_model import LogisticRegression
import multiprocessing
from imblearn.over_sampling import SMOTE
from collections import Counter
import tensorflow as tf
from tensorflow import keras

#Simple MLP NN
# 25000 samples

target_names = None


smote_threshold = 1000 # We have used smote_threshold = 1000 or 3000 for our analysis.

def loadCsv(fileName):
    dataset = pd.read_csv(fileName, sep=",", header=None)
    return dataset

def readCSV(fileName):
    with open(fileName, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    return data

def preprocessing():
    global target_names

    data = loadCsv('dataSetMedium.csv')
    labels = loadCsv('labelsMedium.csv')
    labels.rename(columns={1: 'label'}, inplace=True)
    target_names = list(labels.label.value_counts().index)

    X = data.values
    X = X[4:, 1:] 
    Y = labels.values[:, 1].ravel()

    X_res = X
    Y_res = Y

    # Create dict, where for each class we specify # of samples we desire
    smote_targets = {k: v for k, v in zip(target_names, labels.label.value_counts().values)}

    # Make each class have at least smote_threshold samples
    for k, v in smote_targets.items():
        if v < smote_threshold:
            smote_targets[k] = smote_threshold

    sm = SMOTE(sampling_strategy=smote_targets, k_neighbors=20, random_state=42, n_jobs=multiprocessing.cpu_count() - 1)
    X_res, Y_res = sm.fit_resample(X, Y)

    return X_res, Y_res


dataset, labels= preprocessing()
labels=list(labels)

dic={}
count = 0
labels=np.array(labels)
labels=labels.reshape((30942,1))

for d in labels:
    if d[0] not in dic:
        dic[d[0]]=count
        count+=1

for l in labels:
    l[0]=dic[l[0]]

acc=0
k=10
kf = KFold(n_splits=k,shuffle=True)
for train_index, test_index in kf.split(labels):
    X_train = np.array(dataset[train_index])
    X_test = np.array(dataset[test_index])
    Y_train = np.array(labels[train_index]) 
    Y_test = np.array(labels[test_index])
    
    # create NN model
    model=keras.Sequential([
        keras.layers.Dense(200,activation=tf.nn.sigmoid),
        keras.layers.Dense(200,activation=tf.nn.sigmoid),
        keras.layers.Dense(16,activation=tf.nn.softmax)
        ])

    # Compile NN model
    model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    

    # Fit the NN model
    model.fit(X_train, Y_train, batch_size= 1500, epochs=200)
    
    # evaluate the NN model
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print(test_loss)
    print(test_acc)
    acc+=test_acc

print(acc/k)