import math
import operator
import numpy as np
from operator import itemgetter
from scipy.spatial import distance

class Knn(object):

    def train(self, k, x_train, y_train, distance_string):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train
        self.distance_string = distance_string


    def get_distance(self, instance1, instance2, length):
        if self.distance_string == "euclidean":
            euclidean_distance = np.linalg.norm(instance1 - instance2)
            return euclidean_distance

        elif self.distance_string == "manhattan":
            manhattan_distance = np.linalg.norm(instance1 - instance2, ord = 1)
            return manhattan_distance
        elif self.distance_string == "L_infinite":
            norm_infinite_distance = [abs(a - b) for a, b in zip(instance1, instance2)]

            return max(norm_infinite_distance)


    def predict(self, x_test_data, y_test_data):

       length = len(x_test_data[1:])#feature의 갯수(distance 구할 때 인자로 보내짐)
       predict = []

       for y in range(len(x_test_data[:, 0])):#test data 한 줄 씩(아래로 한 칸씩 반복)
            distance = []
            for x in range(int(self.x_train.shape[0])):#training data column의 개수 만큼 distance 비교
                x_test_instance = x_test_data[y, :]#test data 한줄씩 예측할 것


                dist = self.get_distance(self, self.x_train[x], x_test_instance, length)
                distance.append((dist, self.y_train[x]))

            distance.sort(key=operator.itemgetter(0))
            np_distance = np.asarray(distance)

            neighbor = np_distance[:self.k, 1]

            neighbor_ht = {}
            for element in neighbor:
                if element in neighbor_ht:
                    neighbor_ht[element] += 1
                else:
                    neighbor_ht[element] = 1
            #print(neighbor_ht)
            predict.append(max(neighbor_ht, key=neighbor_ht.get))

       return predict
