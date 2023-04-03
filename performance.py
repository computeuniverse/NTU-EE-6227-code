import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def Confusion_matrix(pred, real):
    C = confusion_matrix(real, pred, labels=[1, 2, 3])
    print(C)
    plt.matshow(C, cmap=plt.cm.Greens)
    plt.colorbar()
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[i, j], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.show()
