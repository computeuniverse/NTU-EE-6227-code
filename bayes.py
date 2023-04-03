import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from math import pi, sqrt
from numpy.linalg import inv, det
from numpy import matmul
from performance import Confusion_matrix


# load data from .mat, datashape is (120, 4) and (120, 1) and (30, 4)
data_train = np.array(loadmat('Data_Train(1).mat')['Data_Train'])
label_train = np.array(loadmat('Label_Train(1).mat')['Label_Train'])
data_test = np.array(loadmat('Data_test(1).mat')['Data_test'])

# divide the dataset into 3 group
g1 = np.empty([40, 4], dtype=float);  index1 = 0
g2 = np.empty([39, 4], dtype=float);  index2 = 0
g3 = np.empty([41, 4], dtype=float);  index3 = 0

for i in range(label_train.shape[0]):
    if label_train[i, 0] == 1:
        g1[index1, :] = data_train[i, :]  # g1 has 40 samples
        index1 = index1 + 1
    elif label_train[i, 0] == 2:
        g2[index2, :] = data_train[i, :]  # g2 has 39 samples
        index2 = index2 + 1
    elif label_train[i, 0] == 3:
        g3[index3, :] = data_train[i, :]  # g3 has 41 samples
        index3 = index3 + 1

# transpose the matrix
g1 = g1.transpose()
g2 = g2.transpose()
g3 = g3.transpose()

# plot every feature's distribution
plt.hist(x=list(g1[0, :]), histtype='stepfilled',bins=50, label='$feature 1$')
plt.hist(x=list(g1[1, :]), histtype='stepfilled',bins=50, label='$feature 2$')
plt.hist(x=list(g1[2, :]), histtype='stepfilled',bins=50, label='$feature 3$')
plt.hist(x=list(g1[3, :]), histtype='stepfilled',bins=50, label='$feature 4$')
plt.legend()
plt.show()

################  Bayes normal distribution ################
# compute the mean vector of every group
g1_mean = np.mean(g1, axis=1)
g2_mean = np.mean(g2, axis=1)
g3_mean = np.mean(g3, axis=1)
print(g1_mean)
print(g2_mean)
print(g3_mean)

# compute covariance matrix
g1_cov = np.cov(g1, bias=True)
g2_cov = np.cov(g2, bias=True)
g3_cov = np.cov(g3, bias=True)
print(g1_cov)
print(g2_cov)
print(g3_cov)

result = []
for u in range(data_test.shape[0]):
    x = np.transpose(data_test[u, :])

    # prior probability
    p_1 = np.exp(-0.5 * matmul(matmul(np.transpose(x - g1_mean), inv(g1_cov)), (x - g1_mean))) / (4*pi*pi*sqrt(det(g1_cov)))
    p_2 = np.exp(-0.5 * matmul(matmul(np.transpose(x - g2_mean), inv(g2_cov)), (x - g2_mean))) / (4*pi*pi*sqrt(det(g2_cov)))
    p_3 = np.exp(-0.5 * matmul(matmul(np.transpose(x - g3_mean), inv(g3_cov)), (x - g3_mean))) / (4*pi*pi*sqrt(det(g3_cov)))

    # posterior probability
    g = max((40/120)*p_1, (39/120)*p_2, (41/120)*p_3)

    # discriminate
    if g == (40 / 120) * p_1:
        result.append(1)
    elif g == (39/120)*p_2:
        result.append(2)
    elif g == (41/120)*p_3:
        result.append(3)
print(result)


# test on train data
result = []
for u in range(data_train.shape[0]):
    x = np.transpose(data_train[u, :])

    # prior probability
    p_1 = np.exp(-0.5 * matmul(matmul(np.transpose(x - g1_mean), inv(g1_cov)), (x - g1_mean))) / (4*pi*pi*sqrt(det(g1_cov)))
    p_2 = np.exp(-0.5 * matmul(matmul(np.transpose(x - g2_mean), inv(g2_cov)), (x - g2_mean))) / (4*pi*pi*sqrt(det(g2_cov)))
    p_3 = np.exp(-0.5 * matmul(matmul(np.transpose(x - g3_mean), inv(g3_cov)), (x - g3_mean))) / (4*pi*pi*sqrt(det(g3_cov)))

    # posterior probability
    g = max((40/120)*p_1, (39/120)*p_2, (41/120)*p_3)

    # discriminate
    if g == (40 / 120) * p_1:
        result.append(1)
    elif g == (39/120)*p_2:
        result.append(2)
    elif g == (41/120)*p_3:
        result.append(3)
print(result)
# performance
Confusion_matrix(pred=result, real=list(label_train))
