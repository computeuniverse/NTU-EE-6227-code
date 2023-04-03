import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# load data from .mat, datashape is (120, 4) and (120, 1) and (30, 4)
data_train = np.array(loadmat('Data_Train(1).mat')['Data_Train'])
label_train = np.array(loadmat('Label_Train(1).mat')['Label_Train'])
data_test = np.array(loadmat('Data_test(1).mat')['Data_test'])

# Create models
tree = DecisionTreeClassifier(random_state=24)
LDA = LinearDiscriminantAnalysis()
Bayes = GaussianNB()

# Train
tree.fit(data_train, label_train)
LDA.fit(data_train, label_train)
Bayes.fit(data_train, label_train)

# Predict
tree_pred = tree.predict(data_test)
LDA_pred = LDA.predict(data_test)
Bayes_pred = Bayes.predict(data_test)
print('tree:', tree_pred, '\n', 'LDA:', LDA_pred, '\n', 'Bayes:', Bayes_pred)