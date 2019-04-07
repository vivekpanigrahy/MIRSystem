from sklearn.model_selection import train_test_split

def getFraction(dataSet, testFraction):
    train, test = train_test_split(dataSet, test_size=testFraction)
    # 80/10/10% split into training, validation and test sets
    #trainData, validateData, testData = np.split(dataSet.sample(frac=1), [int(.8 * len(dataSet)), int(.9 * len(dataSet))])
    return train, test

