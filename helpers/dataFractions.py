# 80/10/10% split into training, validation and test sets
# fractionTrain, fractionTest, fractionVal

import numpy as np
def getFraction(dataSet,labelsData):
    trainData, validateData, testData = np.split(dataSet.sample(frac=1), [int(.8 * len(dataSet)), int(.9 * len(dataSet))])
    trainLabels, validateLabels, testLabels = np.split(labelsData.sample(frac=1), [int(.8 * len(labelsData)), int(.9 * len(labelsData))])
    return trainData.values, validateData.values, testData.values, trainLabels.values.ravel(), validateLabels.values.ravel(), testLabels.values.ravel()