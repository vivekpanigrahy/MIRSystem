import pandas as pd
import csv
def loadCsv(fileName):
	dataset = pd.read_csv(fileName, sep=",", header=None)
	return dataset

def readCSV(fileName):
	with open(fileName, newline='') as csvfile:
		data = list(csv.reader(csvfile))
	return data


