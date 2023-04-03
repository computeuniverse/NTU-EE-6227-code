import numpy as np
from performance import Confusion_matrix
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.linalg import inv
from numpy import matmul, inner
from numpy.linalg import eigh
from math import pow

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
        g1[index1, :] = data_train[i, :]  # group1 has 40 samples
        index1 = index1 + 1
    elif label_train[i, 0] == 2:
        g2[index2, :] = data_train[i, :]  # group2 has 39 samples
        index2 = index2 + 1
    elif label_train[i, 0] == 3:
        g3[index3, :] = data_train[i, :]  # group3 has 41 samples
        index3 = index3 + 1

# transpose the matrix
g1 = g1.transpose()
g2 = g2.transpose()
g3 = g3.transpose()

# compute the mean vector of every group
g1_mean = np.mean(g1, axis=1)
g2_mean = np.mean(g2, axis=1)
g3_mean = np.mean(g3, axis=1)

# compute S_pool
g1_cov = np.cov(g1)
g2_cov = np.cov(g2)
g3_cov = np.cov(g3)
S_pool = (39 * g1_cov + 38 * g2_cov + 40 * g3_cov) / 117

# compute Mahalanobis Distance
def Mahalanobis_D(x, mean1, mean2, mean3, s_pool):
    d_1 = matmul(matmul((x - mean1).reshape(1, -1), inv(s_pool)), (x - mean1).reshape(-1, 1))
    d_2 = matmul(matmul((x - mean2).reshape(1, -1), inv(s_pool)), (x - mean2).reshape(-1, 1))
    d_3 = matmul(matmul((x - mean3).reshape(1, -1), inv(s_pool)), (x - mean3).reshape(-1, 1))

    if min(d_1, d_2, d_3) == d_1:
        return 1
    elif min(d_1, d_2, d_3) == d_2:
        return 2
    elif min(d_1, d_2, d_3) == d_3:
        return 3

# Discriminate observed sample's category
result = []
for u in range(data_test.shape[0]):
    x = data_test[u, :]
    result.append(Mahalanobis_D(x, g1_mean, g2_mean, g3_mean, S_pool))
print('result of Mahalanobis method:', result)



# Although classifier does not need to compute discriminate vectors
# But I do it
G_mean = np.transpose(np.mean(data_train, axis=0))
B = matmul((g1_mean - G_mean).reshape(-1, 1), (g1_mean - G_mean).reshape(1, -1)) + \
    matmul((g2_mean - G_mean).reshape(-1, 1), (g2_mean - G_mean).reshape(1, -1)) + \
    matmul((g3_mean - G_mean).reshape(-1, 1), (g3_mean - G_mean).reshape(1, -1))

# compute matrix W
W1 = 0; W2 = 0; W3 = 0
for i in range(g1.shape[1]):
    x = g1[:, i]
    W1 = W1 + matmul((x - g1_mean).reshape(-1, 1), (x - g1_mean).reshape(1, -1))
for i in range(g2.shape[1]):
    x = g2[:, i]
    W2 = W2 + matmul((x - g2_mean).reshape(-1, 1), (x - g2_mean).reshape(1, -1))
for i in range(g3.shape[1]):
    x = g3[:, i]
    W3 = W3 + matmul((x - g3_mean).reshape(-1, 1), (x - g3_mean).reshape(1, -1))

W = W1 + W2 + W3
E, V = eigh(matmul(inv(W), B))
dis_vector = []
for i in range(len(E)):
    if E[i] > 0:
        dis_vector.append(V[:, i])
print('Discriminate vectors:', dis_vector)
print(B)
print(W)

g1_mapped_1 = matmul(dis_vector[0].reshape(1, -1), g1_mean.reshape(-1, 1))
g1_mapped_2 = matmul(dis_vector[1].reshape(1, -1), g1_mean.reshape(-1, 1))

g2_mapped_1 = matmul(dis_vector[0].reshape(1, -1), g2_mean.reshape(-1, 1))
g2_mapped_2 = matmul(dis_vector[1].reshape(1, -1), g2_mean.reshape(-1, 1))

g3_mapped_1 = matmul(dis_vector[0].reshape(1, -1), g3_mean.reshape(-1, 1))
g3_mapped_2 = matmul(dis_vector[1].reshape(1, -1), g3_mean.reshape(-1, 1))

# Discriminate test data
result = []
for i in range(data_test.shape[0]):
    x = data_test[i, :]
    x_mapped_1 = matmul(dis_vector[0].reshape(1, -1), x.reshape(-1, 1))
    x_mapped_2 = matmul(dis_vector[1].reshape(1, -1), x.reshape(-1, 1))

    Distance1 = pow((x_mapped_1 - g1_mapped_1), 2) + pow((x_mapped_2 - g1_mapped_2), 2)
    Distance2 = pow((x_mapped_1 - g2_mapped_1), 2) + pow((x_mapped_2 - g2_mapped_2), 2)
    Distance3 = pow((x_mapped_1 - g3_mapped_1), 2) + pow((x_mapped_2 - g3_mapped_2), 2)

    min_distance = min(Distance1, Distance2, Distance3)
    if min_distance == Distance1:
        result.append(1)
    elif min_distance == Distance2:
        result.append(2)
    elif min_distance == Distance3:
        result.append(3)
print('result of reducing dimension method:', result)

# Discriminate train data
result = []
for i in range(data_train.shape[0]):
    x = data_train[i, :]
    x_mapped_1 = matmul(dis_vector[0].reshape(1, -1), x.reshape(-1, 1))
    x_mapped_2 = matmul(dis_vector[1].reshape(1, -1), x.reshape(-1, 1))

    Distance1 = pow((x_mapped_1 - g1_mapped_1), 2) + pow((x_mapped_2 - g1_mapped_2), 2)
    Distance2 = pow((x_mapped_1 - g2_mapped_1), 2) + pow((x_mapped_2 - g2_mapped_2), 2)
    Distance3 = pow((x_mapped_1 - g3_mapped_1), 2) + pow((x_mapped_2 - g3_mapped_2), 2)
    min_distance = min(Distance1, Distance2, Distance3)
    if min_distance == Distance1:
        result.append(1)
    elif min_distance == Distance2:
        result.append(2)
    elif min_distance == Distance3:
        result.append(3)
print('result of train data:', result)

# Performance on train data
Confusion_matrix(pred=result, real=list(label_train))


# Plot train data
g1_x = []; g2_x = []; g3_x = []
g1_y = []; g2_y = []; g3_y = []
for j in range(g1.shape[1]):
    x = g1[:, j]
    g1_x.append(inner(dis_vector[0], x.transpose()))
    g1_y.append(inner(dis_vector[1], x.transpose()))

for j in range(g2.shape[1]):
    x = g2[:, j]
    g2_x.append(inner(dis_vector[0], x.transpose()))
    g2_y.append(inner(dis_vector[1], x.transpose()))

for j in range(g3.shape[1]):
    x = g3[:, j]
    g3_x.append(inner(dis_vector[0], x.transpose()))
    g3_y.append(inner(dis_vector[1], x.transpose()))

plt.scatter(g1_x, g1_y, color='blue', s=20, label='$group 1$')
plt.scatter(g2_x, g2_y, color='red', s=20, label='$group 2$')
plt.scatter(g3_x, g3_y, color='green', s=20, label='$group 3$')
plt.xlabel('dis_vector_1')
plt.ylabel('dis_vector_2')
plt.legend()
plt.show()

g1_mean = []




