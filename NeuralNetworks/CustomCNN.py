# CNN with customized dataSet
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

k=10

# load dataset
dataset = np.loadtxt("dataSet.csv", delimiter=",")
labels=pd.read_csv("labels.csv", header=None).values.tolist()

dic={}
count = 0
for d in labels:
	if d[0] not in dic:
		dic[d[0]]=count
		count+=1

for l in labels:
	l[0]=dic[l[0]]

labels=np.array(labels)


acc=0
kf = KFold(n_splits=k,shuffle=True)
for train_index, test_index in kf.split(labels):
    X_train = np.array(dataset[train_index])
    X_test = np.array(dataset[test_index])
    Y_train = np.array(labels[train_index]) 
    Y_test = np.array(labels[test_index])
    X_train = X_train[:,0:506]
    X_test=X_test[:,0:506]
    X_train=X_train.reshape((X_train.shape[0] ,22,23,1))
    X_test=X_test.reshape((X_test.shape[0], 22,23,1))
    input_shape = (22, 23, 1)

    # create CNN model
    model=keras.Sequential([
	keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='sigmoid', input_shape=input_shape),
	keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	keras.layers.Conv2D(64, kernel_size=(5, 5), activation='sigmoid'),
	keras.layers.MaxPooling2D(pool_size=(2, 2)),
	keras.layers.Flatten(),
	keras.layers.Dense(100,activation='sigmoid'),
	keras.layers.Dense(16,activation=tf.nn.softmax)
	])

    # compile the CNN model
    model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    # fit the CNN model
    model.fit(X_train, Y_train, batch_size= 5000, epochs=300)
    
    # evaluate the model
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print(test_loss)
    print(test_acc)
    acc+=test_acc

print(acc/k)


'''
loss: 1.2289 - acc: 0.6057
'''