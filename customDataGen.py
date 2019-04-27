import numpy as np
import pandas as pd
from collections import Counter

features= pd.read_csv('features.csv')
genres= pd.read_csv('genres.csv')
tracks = pd.read_csv('tracks.csv')

genres=np.array(genres)
features=np.array(features)
features=features[3:,:]


tracks=np.array(tracks)
tracks=tracks[2:,:]

l=[]
for g in genres:
    if(g[2]==0):
        l+=[g[3]]

rootTracks={}
for t in tracks:
    if(t[40] in l):
        rootTracks[t[0]]=t[40]
        

trainSet=[]
trainLabels=[]
counter=0
for f in features:
    if(f[0] in rootTracks.keys()):
        counter+=1
        trainSet=trainSet+[list(f[1:].astype(float))]
        trainLabels+=[rootTracks[f[0]]]

trainSet=np.array(trainSet)
trainLabels=np.array(trainLabels)

pd.DataFrame(trainSet).to_csv("dataSet.csv",header=None, index=None)
pd.DataFrame(trainLabels).to_csv("labels.csv",header=None, index=None)