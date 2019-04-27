# simple nn with customized dataSet
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
print(labels.shape)


acc=0
kf = KFold(n_splits=k,shuffle=True)
for train_index, test_index in kf.split(labels):
	X_train = np.array(dataset[train_index])
	X_test = np.array(dataset[test_index])
	Y_train = np.array(labels[train_index]) 
	Y_test = np.array(labels[test_index])
	
	# create NN model
	model=keras.Sequential([
	keras.layers.Dense(60,activation=tf.nn.sigmoid),
	keras.layers.Dense(60,activation=tf.nn.sigmoid),
	keras.layers.Dense(16,activation=tf.nn.softmax)
	])

	# Compile NN model
	model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
	

	# Fit the NN model
	model.fit(X_train, Y_train, batch_size= 1500, epochs=1000)

	# evaluate the NN model
	test_loss, test_acc = model.evaluate(X_test, Y_test)
	print(test_loss)
	print(test_acc)
	acc+=test_acc

print(acc/k)


'''
2 nn, 60, 60, batch=2500, epochs=1000, sigmoid all, adamax, sparse_categorical_crossentropy
loss=1.2524118595077645
acc=0.60695875

2 nn, 60, 60, batch=1500, epochs=1000, sigmoid all, adamax, sparse_categorical_crossentropy
loss=1.2463532319300656
acc=0.60751104


'''

