MAKE KNearestNeighbor
==============
- knn algorith
- without library
- using python

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
![euclidean](https://user-images.githubusercontent.com/37536415/61629686-8feec900-acc0-11e9-9d1c-cb771333e5eb.jpg)

### manhattan 유사도
유클리디언이 제곱 한 후 루트를 씌워 계산을 한다면, 맨하탄은 절댓값으로 계산을 한닥 생각하면 된다.
<img width="262" alt="스크린샷 2019-07-22 오후 8 42 30" src="https://user-images.githubusercontent.com/37536415/61629934-3e930980-acc1-11e9-9557-e4aa330ff2ac.png">

### L_infinite
점 p와 q가 있다면, p의 x좌표와 q의 x좌표를 뺀 절댓값, p의 y좌표와 q의 y좌표를 뺀 절댓값, p의 z좌표와 q의 z좌표를 뺀 절댓값 ....
중 가장 큰 값이다.

## Predict function

```
def predict(self, x_test_data, y_test_data):
```
knn의 predict 함수이다. 
이는 한 raw는 한 개의 item이며, 새로운 값의 class를 계산하고자 할 때, raw 한 개씩 모두 거리를 계산하여, 그 중에서 가장 값이 가까운 것 k개를 뽑아 계산하게 된다. 


## in main.py
main.py는 evaluate 지표들을 직접 만들었다.
accuracy, precision, recall, f-scole, cross validate에 대한 함수 포함되어있다.


