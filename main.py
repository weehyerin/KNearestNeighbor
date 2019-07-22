import numpy as np
import pandas as pd
from knn import Knn

def accuracy(x_test, y_test):
    predict = Knn.predict(Knn, x_test, y_test)
    correct = 0
    miss = 0
    for i in range(len(y_test)):
        if predict[i] == y_test[i]:
            correct += 1
        else:
            miss += 1
    # print(correct)
    # print(miss)
    # print(correct + miss)
    return correct / (correct + miss)

def precision_recall_f(x_test, y_test):
    predict = Knn.predict(Knn, x_test, y_test)

    total_recall = 0
    total_precision = 0
    total_f_measure = 0

    for j in range(10):
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        f_recall = 0
        f_precision = 0
        for i in range(len(y_test)):
            if predict[i] == y_test[i]:
              if predict[i] == j:
                  tp += 1
              elif predict[i] != j:
                 tn += 1
            elif predict[i] != y_test[i]:
              if predict[i] == j:
                fp += 1
              elif predict[i] != j:
                fn += 1
        print(j, "TP : ",tp,"FN : ", fn, "TN : ", tn, " FP : ", fp)
        total_recall += tp / (tp + fn)
        total_precision += tp / (tp + fp)
        f_recall = tp / (tp + fn)
        f_precision = tp / (tp + fp)
        total_f_measure += 2 * ((f_precision * f_recall) / (f_precision + f_recall))


    recall = total_recall / 10
    precision = total_precision / 10
    f_measure = total_f_measure / 10
    # list = []
    # list.append(recall)
    # list.append(precision)
    # list.append(f_measure)
    print("recall : ", recall)
    print("precision : ", precision)
    print("F measure : ", f_measure)
    return

def cross_validation(x_train, y_train, distance, k):

    y = int(x_train.shape[0] / 5)
    sum = 0
    for i in range(5):
        cross_train_x = np.concatenate((x_train[0:y*i, ], x_train[y*(i+1):, ]), axis=0)
        cross_test_x = x_train[y*i:y*(i+1), ]
        cross_train_y = np.concatenate((y_train[0:y*i, ], y_train[y*(i+1):, ]), axis=0)
        cross_test_y = y_train[y*i:y*(i+1), ]
        Knn.train(Knn, k , cross_train_x, cross_train_y, distance)
        sum += accuracy(cross_test_x, cross_test_y)
    return sum / 5




train = pd.read_csv("digits_train.csv", header=None)
test = pd.read_csv("digits_test.csv", header=None)
train = train.values
test = test.values
# print(train.shape)
x_train = train[:, 1:]
y_train = train[:, 0]
x_test = test[:, 1:]
y_test = test[:, 0]
distance = "euclidean"
k = 3
Knn.train(Knn, k, x_train, y_train, distance)
print("test accuracy : ", accuracy(x_test, y_test))
print("train accuracy : ", accuracy(x_train, y_train))
precision_recall_f(x_test, y_test)
print("5 fold cross validation : ", cross_validation(x_train, y_train, distance, k))