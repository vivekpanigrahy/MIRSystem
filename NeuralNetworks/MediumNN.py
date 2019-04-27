# using medium data set
# Simple MLP NN
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# load dataset

k=10

def load_data():
	dataset=pd.read_csv("dataSetMedium.csv", header=None)
	labels=pd.read_csv("labelsMedium.csv", header=None)
	arr=np.array(dataset)
	dataset=arr[4:,1:]
	arr=np.array(labels)
	labels=arr[:,1:]

	dic={}
	count = 0
	for d in labels:
		if d[0] not in dic:
			dic[d[0]]=count
			count+=1

	for l in labels:
		l[0]=dic[l[0]]

	labels=np.array(labels)
	return dataset,labels


dataset, labels=load_data()
acc=0
kf = KFold(n_splits=k,shuffle=True)
for train_index, test_index in kf.split(labels):
	X_train = np.array(dataset[train_index])
	X_test = np.array(dataset[test_index])
	Y_train = np.array(labels[train_index]) 
	Y_test = np.array(labels[test_index])
	
	# create NN model
	model=keras.Sequential([
	keras.layers.Dense(300,activation=tf.nn.sigmoid),
	keras.layers.Dense(300,activation=tf.nn.sigmoid),
	keras.layers.Dense(16,activation=tf.nn.softmax)
	])

	# Compile NN model
	model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
	

	# Fit the NN model
	model.fit(X_train, Y_train, batch_size= 2500, epochs=200)

	# evaluate the NN model
	test_loss, test_acc = model.evaluate(X_test, Y_test)
	print(test_loss)
	print(test_acc)
	acc+=test_acc

print(acc/k)