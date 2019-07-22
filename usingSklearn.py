import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import tree
from sklearn import svm

train = np.genfromtxt("cancer_train.csv", dtype=None, delimiter=',')
test = np.genfromtxt("cancer_test.csv", dtype=None, delimiter=',')
# for line in data:
#     print(line)

neigh = KNeighborsClassifier(n_neighbors=4)
x_train = train[:, 1:]



y_train = train[:, 0]
neigh.fit(x_train, y_train)

x_test = test[:, 1:]
y_test = test[:, 0]

knn_train_pred = neigh.predict(x_train)
knn_test_pred = neigh.predict(x_test)
print("KNN train_accuracy:", metrics.accuracy_score(y_train, knn_train_pred))
print("KNN test_accuracy:", metrics.accuracy_score(y_test, knn_test_pred))

clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
dt_train_pred = clf.predict(x_train)
dt_test_pred = clf.predict(x_test)
print("\nDecision Tree train_accuracy:", metrics.accuracy_score(y_train, dt_train_pred))
print("Decision Tree test_accuracy:", metrics.accuracy_score(y_test, dt_test_pred))

model = svm.SVC(gamma=0.0005, C=100.)
model.fit(x_train, y_train)
svm_train_pred = model.predict(x_train)
svm_test_pred = model.predict(x_test)
list = svm_test_pred
tp = 0
fp = 0
fn = 0
tn = 0
for i in range(len(list)):
    if y_test[i] == list[i]:
        if list[i] == 1:
            tp += 1
        elif list[i] == 0:
            tn += 1
    elif y_test[i] != list[i]:
        if list[i] == 0:
            fn += 1
        elif list[i] == 1:
            fp += 1
# print(y_test)
# print(svm_test_pred)
print("\ntp : ", tp, "tn : ", tn, "fn : ", fn, "fp : ", fp)
precision = tp / (tp+fp)
recall = tp / (tp+fn)
f_measure = 2*precision*recall / (precision + recall)
print("precision : ", precision, "recall : ", recall, "F measure : ", f_measure)
print("SVM train_accuracy:", metrics.accuracy_score(y_train, svm_train_pred))
print("SVM test_accuracy:", metrics.accuracy_score(y_test, svm_test_pred))





