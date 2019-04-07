from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
import helpers.FileReader
import helpers.dataFractions
import pandas as pd

def fullData(gnb, lines,lines2 ):
    data = lines.values
    labels = lines2.values.ravel()
    y_pred = gnb.fit(data, labels).predict(data)
    print("Full Data from train and test Error Rate : %d" % (((labels != y_pred).sum())*100/data.shape[0]))


def dataByFrac(gnb, lines, testFraction):
    trainData, testData = helpers.dataFractions.getFraction(lines, testFraction)
    y_pred = gnb.fit(trainData.iloc[:,:-1], trainData.iloc[:,-1]).predict(testData.iloc[:,:-1])
    print("Test data as Fraction from train Error Rate : %d" % (((testData.iloc[:,-1] != y_pred).sum())*100/testData.shape[0]))

def kFoldData(gnb, lines, nSplits):
    kf = KFold(n_splits=nSplits)
    data = lines
    kf.get_n_splits(data)
    sum = 0
    for train_index, test_index in kf.split(data):
        trainingData, testData = data.iloc[train_index], data.iloc[test_index]
        y_pred = gnb.fit(trainingData.iloc[:,:-1], trainingData.iloc[:,-1]).predict(testData.iloc[:,:-1])
        sum += (testData.iloc[:,-1] != y_pred).sum()
        #print("K Fold mislabeled out of %d points : %d"% (testData.shape[0],(testData.iloc[:,-1] != y_pred).sum()))
    print("K Fold Error Rate : %d" % ((sum/nSplits)*100/testData.shape[0]))


def main():
    gnb = GaussianNB()
    lines = helpers.FileReader.loadCsv('data/dataSet.csv')
    lines2 = helpers.FileReader.loadCsv('data/labels.csv')
    concat_df = pd.concat([lines, lines2], axis=1).reset_index(drop=True)
    #concat_df.sample(frac=1)
    testFraction = 0.6
    nSplits = 5

    fullData(gnb, lines,lines2)
    dataByFrac(gnb, concat_df, testFraction)
    kFoldData(gnb, concat_df,nSplits)

main()