MAKE KNearestNeighbor
==============
## What is KNN?
KNN 알고리즘은 분류나 회귀에 사용되는 패턴 인식 기법이다
이는 lazy algorithm 중 하나로서, 학습하는 데는 시간이 걸리지 않지만, 임의의 값이 어느 class인지 계산해 내는 데에 시간이 오래 걸린다.
knn 알고리즘은 가장 간단한 machine learning 기법이다.

## Calculating distance

```
if self.distance_string == "euclidean":
    euclidean_distance = np.linalg.norm(instance1 - instance2)
    return euclidean_distance

elif self.distance_string == "manhattan":
    manhattan_distance = np.linalg.norm(instance1 - instance2, ord = 1)
    return manhattan_distance
elif self.distance_string == "L_infinite":
    norm_infinite_distance = [abs(a - b) for a, b in zip(instance1, instance2)]

    return max(norm_infinite_distance)

```
knn 알고리즘은 가까운 이웃을 기준으로 class를 분류하기 때문에, 거리 계산이 중요하다.
이 코드에는 거리 계산 방법으로 총 세 가지를 선택할 수 있다.
### Euclidean 유사도
가장 쉽게 이해할 수 있는 거리 계산 법이다. 
단순히, 점 p와 점 q의 직선 거리만 구하면 된다. 
![euclidean](/Users/weehyerin/Desktop/euclidean.gif)

- knn algorith
- without library
- using python
