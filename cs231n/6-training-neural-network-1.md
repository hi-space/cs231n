# 6. Training Neural Network 1

NN의 학습에 대해 알아보자.

![overview](../.gitbook/assets/image%20%28121%29.png)

Activation function 선택, 데이터 전처리, 가중치 초기화, Regularization, gradient checking 등 고려해야 할 요소들이 있다.

## Activation Functions

![activation functions](../.gitbook/assets/image%20%28145%29.png)

다양한 종류의 activation function 과 그들간의 trade-off에 대해서 이야기 해보자

### Sigmoid

![sigmoid](../.gitbook/assets/image%20%2836%29.png)

가장 단순한 Sigmoid 함수. 각 입력을 받아서 그 입력을 \[0, 1\] 사이의 값이 되도록 해준다. 입력 값이 크면 출력은 1에 가까울 거고, 값이 작으면 0에 가까울 것이다. 0 근처 구간은 선형적으로 보인다.

Sigmoid는 뉴런의 firing rate\(점화율\)를 saturation 시키는 것으로 해석할 수 있다. 어떤 값이 0 ~ 1사이의 값을 가지면 이를 firing rate 라고 생각할 수 있다. ReLU가 생물적 타당성이 크긴 하지만, sigmoid 또한 그런 역할을 한다는 걸 알고가자.

![problems of sigmoid](../.gitbook/assets/image%20%28297%29.png)

하지만 Sigmoid에는 몇가지 문제점이 있다.

1. Saturated neurons "kill" the gradients
2. Sigmoid outputs are not zero-centered
3. exp\(\) is a bit compute expensive

하나하나 알아보자.

![](../.gitbook/assets/image%20%28164%29.png)

**Saturation 되는게 gradient를 없앤다. \(Saturated neurons "kill" the gradients\)**

Gradient descent를 한다고 보자. 이런 값들이 계속 연쇄적으로 backprop 될거다. 

* X = -10 일 때, gradient = 0이다. 음의 큰 값이면 sigmoid가 flat 하게 되고 gradient가 0이 될 것이다. 거의 0에 가까운 값이 backprop이 될 거다. 이 부분에서 gradient가 죽어버리게 되고 밑으로 0이 계속 전달되게 되어 학습이 잘 이루어지지 하는다.
* X = 0 일 때,  이 구간은 linear 하기 때문에 잘 동작할 거다. 그럴싸한 gradient를 얻게 될 거다. 
* X = 10 일 때,  큰 양수일 경우에도 sigmoid가 flat 하기 때문에 gradient 들이 다 죽게 된다. 역시나 학습이 잘 되지 않을 거다.

![](../.gitbook/assets/image%20%28297%29.png)

**sigmoid 출력이 zero-centered 하지 않다. \(Sigmoid outputs are not zero-centered\)**

뉴런의 입력이 항상 양수이거나 항상 음수일 경우, dL/df\(활성함수\)가 넘어오면 Local gradient 는 이 값이랑 곱해질 것이고, df\(활성함수\)/dW는 그냥 X가 될 것이다. 그렇게 되면 gradient 부호는 위에서 내려온 gradient 의 부호와 같아질 거다. 이는, W가 모두 같은 방향으로만 움직일 것임을 의미한다. 파라미터를 업데이트 할 때 다 같이 증가하거나 다 같이 감소하거나 할 수 밖에 없다. 이런 gradient 업데이트는 비효율적이다.

![](../.gitbook/assets/image%20%28256%29.png)

위의 예제로 봤을 때, gradient가 이동할 수 있는 방향은 4분면 중 이 두 영역 밖에 안될거다. 두 방향으로 밖에 gradient가 업데이트 되지 않는다. 가장 최적의 W 업데이트가 파란색 화살표고, 초기 시작점으로부터 내려간다고 할 때, gradient는 파란색 방향으로 내려갈 수 없다. 그 방향으로는 움직일 수 없기 때문에 여러 번의 gradient를 수행해야 한다. 이게 바로 우리가 일반적으로 zero-mean data를 원하는 이유다. 입력 X가 양수/음수를 모두 가지고 있으면 전부 같은 방향으로 움직이지는 않을거다.  

**Exp\(\)로 인해 계산 비용이 크다 \(exp\(\) is a bit compute expensive\)** 

마이너한 문제이긴 한데 굳이 문제를 꼽자면 그렇다는 거. 오히려 내적이 더 비용이 비싸다.

### tanh

![Tanh](../.gitbook/assets/image%20%2862%29.png)

Sigmoid와 비슷하지만 범위가 \[-1, 1\]이다. Zero-centered 됐지만 여전히 saturation 때문에 gradient가 죽는다. Gradient가 평평해지는 구간이 있다.

### ReLU \(Rectified Linear Unit\)

![ReLU](../.gitbook/assets/image%20%2846%29.png)

Element-wise 연산을 수행하며 입력이 음수면 값이 0이 되고, 양수면 입력값 그대로를 출력한다. 가장 일반적으로주 사용되는 activation function이다.

양의 값에서는 linear 하기 때문에 saturation 되지 않는다. 적어도 영역의 절반은 saturation 되지 않는다.

그리고 함수가 단순하기 때문에 \(단순 max 연산\) 계산이 빠르다. Sigmoid, tanh 보다 수렴속도가 거의 6배 정도 더 빠르다. 

생물학적 타당성도 훨씬 더 잘 맞는다. 실제 뉴런을 관찰해보면 sigmoid 보다는 ReLU 스럽다. AlexNet이 처음 ReLU를 사용하기 시작했다.

![problem of ReLU](../.gitbook/assets/image%20%28251%29.png)

하나 문제가 있다면 zero-centered 하지 않다는 거다. 또 양의 구간\(regime\)에서는 saturation 되지 않지만 음의 regime에서는 여전히 saturation 된다.

* X = -10 일 때, gradient는 0
* X = 10일 때, Linear 영역이기 때문에 잘 동작
* X = 0일 때, gradient는 0

ReLU는 gradient의 절반을 죽여버리는 꼴이다.

![dead ReLU](../.gitbook/assets/image%20%28177%29.png)

Dead LeLU 문제가 있다. 초록색, 빨간색 선분을 각 ReLU라고 봤을 때, ReLU가 data cloud에서 떨어져 있는 경우에 dead ReLU가 발생할 수 있다. Dead ReLU 에서는 activate가 일어나지 않고 gradient가 update 되지 않는다. active ReLU는 일부는 active되고 일부는 active 하지 않을 거다.

이는 초기화를 잘못한 경우에 발생하는 문제다. 가중치 평면이 data cloud에서 멀리 떨어져 있는 것이 문제다. 이 경우 데이터 입력에서부터 activate 되지 않을거고 backprop도 안될거다. 전혀 Update가 안되는 거다.

더 흔한 문제는 learning rate가 지나치게 높은 경우다. Update를 지나치게 크게 해버려서 가중치가 날뛴다면 ReLU가 데이터의 manifold를 벗어나게 된다. 처음에는 학습이 잘 되다가 갑자기 update 되지 않는 경우가 이런 문제가 발생했을 때 이다.

실제로 학습을 다 시켜놓은 네트워크를 보면 10~20퍼는 dead ReLU가 되어 있다. ReLU를 사용하고 있따면 대부분의 네트워크가 이 문제를 겪는다. 뭐 근데 그 정도가 네트워크 학습에 크게 지장이 있진 않다.

실제로 ReLU를 초기화 할 때 positive biases를 추가해주는 경우가 있다. update시에 active ReLU가 될 가능성을 높이는 거다. 이게 도움이 된다는 의견도 있고 그렇지 않다는 사람도 있어서 항상 이 방법을 사용하지는 않고, 대부분은 zero-bias로 초기화 한다.

### Leaky ReLU

![](../.gitbook/assets/image%20%28211%29.png)

ReLU를 개선한 버전이 나왔다. ReLU와 유사하지만 negative regime 에서 더 이상 0이 아니다. negative에도 기울기를 살짝 주면서 앞서 말한 문제를 해결했다. 

Negative space에서도 saturation 하지 않고, 여전히 계산이 효율적이다. Dead ReLU 현상도 더 이상 없다.

### PReLU \(Parametric Rectifier\)

![PReLU](../.gitbook/assets/image%20%28238%29.png)

PReLU는 Leaky ReLU와 유사하지만 negative regime의 기울기가 alpha라는 파라미터로 정의 된다. alpha\(기울기값\)를 정해놓는 것이 아니라 backprop으로 학습시키는 파라미터로 만듦으로써 활성함수가 좀 더 유연해 질 수 있다.

### ELU \(Exponential Linear Units\)

![ELU](../.gitbook/assets/image%20%28215%29.png)

ELU는 ReLU의 이점을 그대로 가져오고\(saturation 되는 구간이 있다는 점도\), Leaky ReLU나 PReLU 처럼 zero-mean에 가까운 출력값을 보인다. negative regime 에서 기울기를 가지는 것 대신 다시 saturation되는 걸 볼 수 있다. 이 Saturation이 좀 더 noise에 강인할 수 있다는 아이디어에 기안했다. 이러한 deactivation 이 모델에 robust함을 더 할 수 있다고 논문에서는 주장한다. 

### Maxout

![Maxout](../.gitbook/assets/image%20%28137%29.png)

Maxout은 여태 본 활성함수와는 좀 다르다. 입력을 받아드리는 특정한 기본 형식을 미리 정의하지 않는다. 대신 \(w1와 x를 내적한 값 + b1\)과 \(w2 와 x를 내적한 값 + b2\)중 max 값을  뽑는다. 두개의 선형함수를 취한다는 점에서 ReLU와 비슷하다. 이 함수도 linear 하기 때문에 saturation 되지 않을 거다.

W1, W2를 모두 지니고 있어야 하기 때문에 뉴런당 파라미터 수가 두배가 된다는 문제는 있다.

![](../.gitbook/assets/image%20%28294%29.png)

실제로 가장 많이들 쓰는 것은 ReLU다. 다양한 문제에서 잘 동작하는 편이다. 다만 learning rate를 섬세하게 잘 결정해야 한다. Leaky ReLU, Maxout, ELU 같은 activation function은 아직 실험단계이긴 하지만 써볼 수 있다. 문제에 따라 어떤 activation function을 사용할 지 다를 거다. tanh나 sigmoid도 써볼 수 있겠지만 대게는 ~LU \(ReLU의 변종\)들이 좋다. 

## Data Processing

실제 네트워크를 트레이닝 시켜보자.

![](../.gitbook/assets/image%20%28300%29.png)

가장 대표적인 전처리 과정은 zero-mean으로 만들고 normalize 하는 거다. normalization은 보통 표준편차로 한다. Normalization을 해주는 이유는 모든 차원이 동일한 범위 안에 있게 해줌으로써 전부 동등한 contribute을 할 수 있도록 만들기 위함이다.  

실제로는 이미지의 경우는 전처리로 zero-centering 정도만 하고 Normalization 하지 않는다. 이미지는 이미 각 차원 간에 스케일이 어느 정도 맞춰져 있기 때문이다. 스케일이 다양한 ML 문제와는 달리 이미지에서는 normalization을 엄청 잘 해줄 필요는 없다.

![](../.gitbook/assets/image%20%2887%29.png)

PCA나 whitening 등의 복잡한 전처리 방법도 있지만 이미지에서는 단순히 zero-mean 정도만 사용한다. 일반적으로 이미지를 다룰 때는 굳이 입력을 더 낮은 차원으로 projection 시키지 않는다. CNN에서는 원본 이미지 자체의 spatial 정보를 이용해 이미지의 spatial structure를 얻을 수 있도록 한다.

![](../.gitbook/assets/image%20%2815%29.png)

평균값은 보통 전체 Training data에서 계산한다. 입력 이미지의 사이즈를 서로 맞춰주는데 네트워크에 들어가기 전에 평균값을 빼주게 된다. Test data도 그 평균값으로 빼준다. 일부 네트워크는 채널 전체의 평균을 구하지 않고 채널마다 평균을 독립적으로 계산하는 경우도 있다. \(ex, VGGNet\)

mini-batch 단위로 training 시키는 경우도 전체 training data set의 평균으로 한다. batch에서 뽑은 데이터도 사실 전체 데이터에서 나온 것이고, 이상적으로 생각해보면 결국은 batch 평균이나 전체 평균이나 유사할 거다. 그러니 처음에 한번만 구하는게 낫다. 반대로 엄청 큰 데이터 전체 평균을 구할때도 굳이 다 쓰지 않고 적절히 샘플링해서 구할 수도 있을 거다.

