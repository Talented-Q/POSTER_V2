import itertools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# -*- coding:utf-8 -*-

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label',fontsize=12)
    plt.xlabel('Predicted Label',fontsize=12)
    plt.show()



cnf_matrix = np.array([[ 299 ,   6 ,   5 ,   3 ,   1 ,   4,   11],
 [   9,   51   , 0,    2   , 8,    2   , 2],
 [   2 ,   1  ,120 ,   6   ,13 ,   9  ,  9],
 [   5  ,  1   , 7 ,1148   , 2  ,  4 ,  18],
 [   0   , 0  ,  9  ,  4  ,442   , 1  , 22],
 [   2    ,0 ,   7   , 3 ,   0  ,145 ,   5],
 [  10    ,0,    6   ,11,   29   , 0,  624]])

class_names = ["SU", 'FE', 'AN', 'HA', 'SA', 'DI', 'NE']


plt.figure(dpi=200)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=None)
