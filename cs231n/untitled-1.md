---
description: Image Classification
---

# 2. Image Classification

## Image Classification

![Image Classification](../.gitbook/assets/image%20%28288%29.png)

Image Classification은 컴퓨터 비전 분야에서 Core Task에 속한다. Input Image를 받고, 시스템에서 미리 정해놓은 카테고리 내에서 어떤 카테고리에 속하는 지 판단하는 문제이다.

![Semantic Gap](../.gitbook/assets/image%20%28239%29.png)

Semantic Gap : 고수준 언어의 언어 요소와 이들을 실현하기 위한 컴퓨터의 기능 사이에는 큰 격차가 있는 것. 

이미지 데이터는 Semantic Gap이 있다. 컴퓨터에게 이미지는 큰 격자모양의 숫자집합으로 밖에 보이지 않을 거다. "고양이" 라는 label은 우리가 이 이미지에 붙힌 의미상의 label이다. 이 이미지에 아주 미묘한 변화만 주더라도 픽셀 값은 크게 변할 것이다.

* Viewpoint variation
* Illumination : 이미지가 밝을 수도, 어두울 수도 있다.
* Deformation : 객체 자체에 변형이 있을 수 있다.
* Occlusion : 객체의 일부가 가려져 있을 수 있다.
* Background Clutter : Object가 배경과 유사하게 생겼을 수도 있다.
* intraclass variation : 하나의 클래스에도 다양성이 존재하는데, 하나의 개념으로 그 객체의 다양한 모습들을 담을 수 있어야 한다.

위와 같은 문제들에서도 robust 한 알고리즘을 만들어야 한다.

### Hard-code algorithm

![hard-code algorithm](../.gitbook/assets/image%20%28117%29.png)

Object를 인식하는 건 직관적이고 명시적인 알고리즘이 존재하지 않는다. input으로 image를 받고 ouput으로 class\_label 값을 리턴하는 함수를 만든다고 했을 때 어떤 내용이 채워져야 할까?

![hard-code algorithm](../.gitbook/assets/image%20%28278%29.png)

알고리즘을 만든다고 했을 때 이런 방법이 있을 수 있다. 이 이미지에서 Edges를 계산하고 Corners 들을 뽑는다. 그리고 이 Corner에 뾰족한 모양의 귀가 있고 다른 Corner의 어느 위치 쯤에는 코가 있을 거고, 이런 방식으로 Object의 특징들을 써내려 갈 수 있을 거다. 인식을 위한 "명시적인 규칙 집합"을 써내려 가는 방법이다. 

하지만 이런 알고리즘은 Robust 하지 않다. 또 다른 Object를 인식해야 한다면 또 그 객체에 대한 특성을 나열하며 알고리즘을 재정의 해야 한다. 가령 10개의 Object를 인식하기 위해선 10개의 각각의 알고리즘이 필요한 것이다. 이 세상에 존재하는 다양한 Object들에게 유연하게 적용이 가능한 확장성 있는 알고리즘이 필요하다.

### Data-Driven Approach

![Data-Driven Approach](../.gitbook/assets/image%20%28205%29.png)

그를 위한 하나의 insight는 Data Driven Approach다. 직접 규칙을 나열하는 것 대신에 많은 데이터들을 수집하는 것이다. 많은 데이터셋들을 모아서 ML 알고리즘을 돌리게 되면,  ML 알고리즘은 어떤 식으로든 데이터를 잘 요약해서 다양한 Object들을 인식할 수 있는 모델을 만들어 낼 것이다.

이를 구현하기 위해서는 이제 Train 함수와 Predict 함수 두가지가 필요할 거다. Train 함수에서는 input image와 label 값을 input으로 받아 training을 하고 training된 model를 output으로 가질 것이다. Predict 함수에서는 Train 함수에서 만들어진 model과 알고자 하는 image를 input으로 받아서 해당 image가 어떤 class인지 model로 예측 된 label 값을 output으로 한다. 이게 바로 ML의 key insight이고 여태껏 잘 사용해 온 방법이였다.

## Nearest Neighbor

![Nearest Neighbor](../.gitbook/assets/image%20%28243%29.png)

딥러닝으로 넘어가기 전에 심플한 Classifier 부터 살펴보자.  Nearest Neighbor 이다. Data-driven approach로서 기본적인 알고리즘이고 위에서 말했듯이 두가지의 함수가 필요하다.

* Train Step : 모든 데이터와 그 데이터에 해당하는 label을 저장한다.
* Predict Step : 새로운 이미지가 들어오면 기존의 학습 데이터와 비교해서 가장 유사한 이미지로 label을 한다.

![CIFAR 10](../.gitbook/assets/image%20%28138%29.png)

이 이미지에 Nearest Neighbor 알고리즘을 적용하면 Training set에서 가장 가까운 샘플을 찾게 된다. 이 때 Test image와 학습 이미지들 간에 비교를 할 때 어떤 비교 함수\(Distance Metric\)를 사용하는 지에 따라 결과가 상이할 수 있다.

### Distance Metric

![L1distance](../.gitbook/assets/image%20%2839%29.png)

가장 심플한 Distance Metric은 L1 distance \(Manhattan distance\). Pixel-wise로 비교를 하는데,  test image와 training image의 같은 위치의 픽셀 간 차이 값을 계산하는 것이다. 딱 봐도 Image Classification 에서는 좋지 않은 방법임을 알 수 있다.

### Code

> ```python
> import numpy as np
>
> class NearestNeighbor(object):
>   def __init__(self):
>     pass
>
>   def train(self, X, y):
>     """ X is N x D where each row is an example. Y is 1-dimension of size N """
>     # the nearest neighbor classifier simply remembers all the training data
>     self.Xtr = X
>     self.ytr = y
>
>   def predict(self, X):
>     """ X is N x D where each row is an example we wish to predict label for """
>     num_test = X.shape[0]
>     # lets make sure that the output type matches the input type
>     Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
>
>     # loop over all test rows
>     for i in xrange(num_test):
>       # find the nearest training image to the i'th test image
>       # using the L1 distance (sum of absolute value differences)
>       distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
>       min_index = np.argmin(distances) # get the index with smallest distance
>       Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
>
>     return Ypred
> ```

* train : Memorize training data
* predict : For each test image, Find closest train image and Predict label of nearest image

Q. 만약 N개의 example을 돌릴 때 training, prediction은 각각 시간복잡도가 어떻게 될까?  
A. Train : O\(1\), Predict O\(N\)

Train 속도는 나쁘지 않지만 Predict 하는 것은 모든 이미지 마다 비교를 해야하기 때문에 꽤 느린 작업이다\(Train Time &lt; Test Time\). 하지만 보통 Train time이 좀 느려도 되지만 Test Time은 빠르길 원한다. CNN 같은 parametic model은 train time 이 좀 걸릴 수 있겠지만 test time은 만들어져 있는 parameter를 사용하기 때문에 빠르다.

### Limitation

![Decision Region of Nearest Neighbor](../.gitbook/assets/image%20%28192%29.png)

각 점은 학습 데이터, 색은 클래스 label 이다. 2차원 평면 내의 모든 좌표에서 각 좌표가 어떤 학습 데이터와 가장 가까운지 측정한다. Nearest Neighbor 는 이렇게 공간을 나눠 각 레이블로 분류한다. 여기서 문제를 볼 수 있다.

* 대부분이 초록색 점인데, 중간에 노란 점 하나가 있다. 가장 가까운 이웃만을 보기 때문에 녹색 클래스 가운데에 노란 클래스가 위치 할 수 있다. 이 점은 녹색이어야 맞다.
* 초록색의 한 점이 파란색 영역에 삐죽 침범되어 있다. 이게 noise다.

## K-Nearest Neighbors

![K-nearest Neighbors](../.gitbook/assets/image%20%2821%29.png)

위와 같은 문제들을 해결하기 위해 K-Nearest Neighbors가 나왔다. 단순하게 가장 가까운 이웃만 찾지 않고 Distance metric을 이용해 가까운 이웃을 K개 만큼 찾고 이웃끼리 투표를 하는 방식이다. 투표를 하는 방법도 다양하다. 예를 들면 거리별 가중치를 고려하는 것과 같은 규칙이 있을 수 있다. 가장 잘 동작하기 위한 심플한 방법은 득표수로만 판단하는 것이다. \(majority vote\)

K 값이 증가함에 따라 noise가 정리되고 경계도 많이 부드러워진 것을 확인 할 수 있다. K 값이 커질 수록 noise들에 대해 좀 더 Robust 해질 수 있을 거다. 그럼 그 서로 다른 점들은 어떻게 비교하는게 좋을까?

### Distance Metric

![Distance Metric : L1, L2](../.gitbook/assets/image%20%2859%29.png)

L1 \(Manhattan\) Distance : 픽셀 간 차이의 절대값의 합  
L2 \(Euclidean\) Distance : 픽셀 간 차이의 제곱합의 제곱근

어떤 Distance Metric을 선택하는 것도 중요한 문제다. L1 Distance 같은 경우는 어떤 좌표 시스템이냐에 따라 많은 영향을 받는다. 기존의 좌표계를 회전시키면 L1 Distance 값들은 변하게 되지만 L2 Distance는 원형이라서 좌표계와 아무 연관이 없다. feature vector의 각각 요소가 개별적인 특징적 의미가 있다면 L1 Distance가 잘 맞겠지만, feature vector가 일반적인 벡터이고 요소들 간의 실질적인 의미를 잘 모르는 경우에는 L2 Distance가 더 맞을 수 있다. 

![K-NN Distance Metric](../.gitbook/assets/image%20%2882%29.png)

Distance Metric에 따라 Decision Region이 달라진 것을 볼 수 있다. L1은 Decision Region이 좌표 축에 영향을 받는 것이 보이고, L2는 좌표 축의 영향을 받지 않아서 영역 모양이 더 자연스럽다. 

### Hyperparameters

![Hyperparameter](../.gitbook/assets/image%20%28178%29.png)

k-NN에서 k와 Distance metric은 Hyperparameter이다. Train time에 학습시키는 것이 아니라 미리 사전에 정해줘야 한다. 어떻게 hyperparameter 값을 정하는 지는 problem-dependent 하다. 다양한 hyperparameter 값을 시도해보고 적합한 값을 찾아야 한다.

![Setting Hyperparameters](../.gitbook/assets/image%20%28260%29.png)

![Setting Hyperparameter](../.gitbook/assets/image%20%28267%29.png)

hyperparameter를 세팅하기 위한 몇가지 아이디어가 있다.

* Idea1 
  * 학습 데이터의 정확도와 성능을 최대화 하는 hyperparameter를 선택

    -&gt; 학습데이터는 잘 맞출 수 있겠지만 새로운 데이터가 왔을 때는 잘 못 맞출 것이다.
* Idea2
  * 다양한 train data로 학습시키고 test data에 적용했을 때 가장 성능이 좋은 hyperparameter 선택

    -&gt; 이것도 여전히 새로운 데이터가 들어왔을 때 잘 맞출 수 없을 것이다. 단지 test set에서만 잘 동작하는 hyperparameter를 고르게 될 것이고, 이게 새로운 데이터에 대한 성능을 대표할 수는 없다.
* Idea3 
  * training data set, validation set, test set 세가지로 데이터를 나눈다.
  * training data set : 다양한 hyperparameter를 훈련시킨다.
  * validation set : 검증을 한다. 가장 좋았던 hyperparameter를 선택한다.
  * test set : 최종적으로 개발/디버깅을 마친 후 validation 에서 가장 좋았던 분류기를 가지고 딱 한번 테스트 한다. 그게 바로 이 모델의 성능이 될 거다.
* Idea4 \(Cross Validation\)
  * training/validation data set을 나눌 때 더 잘게 쪼개서 번갈아가며 validation set으로 선택한다. test data set은 미리 정해놓고 마지막에만 쓴다. 
  * 예를 들어 1, 2, 3, 4 fold에서 training 시키고 5로 validation 한다. 그리고 1, 2, 3, 5 fold에서 training 시키고 4 fold로 validation 한다. 이런식으로 순환한다.
  * 표준이긴 하지만 DL과 같은 큰 모델을 학습시킬 때는 학습 자체가 계산량이 너무 많아서 실제로는 잘 쓰지 않는다. 작은 데이터셋일 경우에만 사용한다.
  * test set이 알고리즘 성능 향상에 미치는 영향을 알고자 하면 이런 방법을 쓰면 좋다. 여러 validation folds 별 성능의 variance를 알 수 있게 된다. 어떤 hyperparameter가 가장 좋은 지 알 수 있을 뿐만 아니라 그 성능의 variance도 알 수 있다.

Q. training set 과 validation set의 차이는?  
A. training set에서는 분류기를 학습시킨다. 그리고 validation set을 가져와서 training set과 비교한다. validation set에는 분류기가 얼마만큼의 정확도가 나오는지 판단하는 set 이다. training set은 label을 볼 수 있지만 validation set은 label을 볼 수 없다. 단지 알고리즘이 얼마나 잘 동작하는지 확인할 때에만 사용하는 것이기 때문.

### Limitation

![](../.gitbook/assets/image%20%285%29.png)

K-NN은 input이 image 일 때는 사용하지 않는다. 너무 느리기도 하고 L1, L2와 같은 Distance metric이 이미지 간 거리를 측정하기에는 적절하지 않기 때문이다. 위의 예시에서 하나의 이미지를 boxed, shifted, tinted 했지만 모두 동일한 L2 distance를 갖는다. 눈으로 봐도 극명히 다른 이미지인데 컴퓨터는 같은 이미지로 판단한다는 의미이다. 

![curse of dimensionality](../.gitbook/assets/image%20%28193%29.png)

또 하나의 문제는 차원의 저주. K-NN은 결국 공간을 나누는 알고리즘이다. 잘 동작하기 위해선 충분한 training set이 필요하다. 그 필요한 training set이 차원이 증가함에 따라 기하급수적으로 증가하게 된다. 고차원의 이미지라면 모든 공간을 조밀하게 메울 만큼의 데이터가 필요하다.

## Linear Classifier

![Linear Classifier](../.gitbook/assets/image%20%28151%29.png)

Linear Classifier는 parametric model의 기본적인 구조이다. 

X : input image  
W \(or theta\) : 가중치 파라미터  
f 함수 : data X, parameter W를 통해 10개 클래스의 score를 output으로 낸다.

K-NN은 파라미터가 없었다. 단순히 데이터 뭉치를 넣어서 판단할 뿐이였다. 하지만 parametric approach에서는 training data의 정보를 요약해 파라미터 W에 몰아준다. 그럼 test time에는 training data가 없어도 그 정보들을 요약한 W를 이용하면 된다. 

![Linear Classifier](../.gitbook/assets/image%20%2870%29.png)

이제 이 Function을 어떻게 정의하고 Parameter를 어떻게 정의할 것이냐 그 조합을 고민하는 것이 딥러닝의 문제이다. 단순한 방법은 x와 W를 그냥 곱하는 것이다. bias term을 더해주기도 한다. 데이터와 무관하게 특정 클래스에 priority를 부여하는 역할을 한다. 예를 들어 고양이와 개를 판단하는 문제에서 고양이의 데이터가 너무 많으면 고양이 클래스의 bias 값을 높게 설정해준다.

![Example of Linear Classifier](../.gitbook/assets/image%20%28126%29.png)

W = 4 x 3 =&gt; 픽셀이 4개고 class가 3개  
Linear Classification은 템플릿 매칭과 거의 유사하다. 가중치 행렬 W의 각 행은 각 이미지에 대한 템플릿으로 볼 수 있고, 그 행 벡터와 이미지의 열 벡터간의 내적을 계산하는데, 여기서 내적은 결국 클래스 간 템플릿의 유사도를 측정하는 것과 유사하다는 걸 알 수 있다. bias 는 데이터와는 독립적으로 scailing offsets를 더해주는 것이다.

![Linear Classifier : Three Viewpoints](../.gitbook/assets/image%20%2899%29.png)

Algebraic Viewpoint로 봤을 땐 저 공식으로 계산해서 나오는 score로 class를 결정하는 것이고, Visual Viewpoint와 Geometric Viewpoint로도 볼 수 있다.

Visual Viewpoint로 보면 하나의 클래스 당 하나의 템플릿이 나오는 것을 볼 수 있다. 이게 Linear Classifier의 문제 중 하나이다. 한 클래스 내에 다양한 특징들이 존재할 수 있지만, 모든 것들을 평균화 시키기 때문에 다양한 모습들이 있더라도 각 카테고리를 인식하기 위한 템플릿은 단 하나밖에 없다.

Geometric Viewpoint로 보자. 이미지를 고차원 공간의 한 점으로 보는 것이다. 그리고 Linear Classifier가 각 클래스를 구분시켜주는 선형 결정 경계를 그어주는 역할을 한다. 이 관점으로 해석했을 때는 아래와 같은 문제가 있다.

### Limitation

![](../.gitbook/assets/image%20%28132%29.png)

* Parity Problem : 홀/짝수를 분류하는 것과 같은 반전성 문제
* Multiimodal Problem : 한 클래스가 다양한 공간에 분포할 수 있는 경우에 단순히 Linear 직선을 그어 문제를 해결 할 수가 없다.

