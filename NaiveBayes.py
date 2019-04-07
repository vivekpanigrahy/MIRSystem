from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
import helpers.FileReader
import helpers.dataFractions

def fullData(gnb, lines,lines2 ):
    data = lines.values
    labels = lines2.values.ravel()
    y_pred = gnb.fit(data, labels).predict(data)
    print("Full Data mislabeled points out of %d points : %d"% (data.shape[0],(labels != y_pred).sum()))

def data80_10_10frac(gnb, lines,lines2):
    trainData, validateData, testData, trainLabels, validateLabels, testLabels = helpers.dataFractions.getFraction(lines,lines2)
    y_pred = gnb.fit(trainData, trainLabels).predict(testData)
    print("Ratio mislabeled points out of %d points : %d"% (testData.shape[0],(testLabels != y_pred).sum()))

def kFoldData(gnb, lines,lines2):
    kf = KFold(n_splits=3)
    data = lines.values
    labels = lines2.values.ravel()
    kf.get_n_splits(data)
    for train_index, test_index in kf.split(data):
        trainingData, testData = data[train_index], data[test_index]
        trainingLabel, testLabel = labels[train_index], labels[test_index]
        y_pred = gnb.fit(trainingData, trainingLabel).predict(testData)
        print("K Fold mislabeled out of %d points : %d"% (testData.shape[0],(testLabel != y_pred).sum()))

def main():
    gnb = GaussianNB()
    lines = helpers.FileReader.loadCsv('data/dataSet.csv')
    lines2 = helpers.FileReader.loadCsv('data/labels.csv')
    fullData(gnb, lines,lines2)
    data80_10_10frac(gnb, lines,lines2)
    kFoldData(gnb, lines,lines2)

main()