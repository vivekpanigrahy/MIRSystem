import pickle
import matplotlib.pyplot as plt


accuracies = None
num_estimators = [10, 25, 50, 75, 100, 125, 150, 200, 600]
output_folder = 'adaboost_outputs'


with open(output_folder + '/accuracies_3000.pickle', 'rb') as handle:
    accuracies = pickle.load(handle)
    accuracies.append(0.3091)

plt.plot(num_estimators, accuracies, 'go--')
plt.legend(loc='best')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('AdaBoost (SMOTE_3000) Accuracy For Varying # Weak Learners')
plt.show()