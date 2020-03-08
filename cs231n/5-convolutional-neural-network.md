# 5. Convolutional Neural Network

## History

![Mark I Perceptron](../.gitbook/assets/image%20%28175%29.png)

1957년. Frank Rosenblatt가 Mark I Perceptron machine을 개발했다. Perceptron을 구현한 최초의 기계였다. 출력값은 0 과 1 뿐이였지만 여기에도 가중치 W를 Update하는 Rule이 존재했다. 이는 지금의 Backprop과 유사하다고 볼 수 있다. 물론 그 때는 Backprop 개념이 없었기 때문에 W 값을 이리저리 조절하면서 맞추는 식이였다.

![Adaline/Madaline](../.gitbook/assets/image%20%28257%29.png)

1960년. Widrow, Hoff가 Adaline and Madaline을 개발했다. 최초의 Multilayer Perceptron Network였다. 이 때도 여전히 Backprop 같은 학습 알고리즘은 여전히 없었다.

![First back-propagation](../.gitbook/assets/image%20%28254%29.png)

Backprop은 Rumelhart가 1986년 처음 만들었다. 지금과 같은 형태처럼 Chain rule과 update rule을 볼 수 있다. 이 때 최초로 network 학습 개념이 정립되기 시작했다. 하지만 이 이후로는 NN에 대해 새로운 네트워크 발견이나 사용처를 찾지 못해 한동안 잠잠했다.

![Hinton and Salakhutdinov](../.gitbook/assets/image%20%28156%29.png)

그리고 2000년대가 되어서야 다시 연구에 활기를 찾았다. 위의 논문에서 DNN 학습 가능성을 봤고 꽤 효과적이였다. 하지면 어전히 모던한 방법은 아니였다. backprop 하기 위해서 전처리가 필요했다. 초기화를 위해 RBM을 이용해 Hidden Layer의 W를 학습시켰다. 그리고 이렇게 초기화 된 Hidden Layer를 이용해 전체 신경망을 backprop 하거나 fine tune 했다.

![First strong result](../.gitbook/assets/image%20%28129%29.png)

실제로 NN에 다시 집중된 건 2012년이다. 음성인식 분야에서 좋은 성능을 보였다. 또 영상인식에 관한 논문도 나왔다. Imagenet Classification 에서 최초로 NN을 사용했고 성능이 좋았다. AlexNet은 ImageNet benchmark의 error를 극적으로 감소시켰다. 그 이후로 ConvNets은 널리 사용되고 있다.

## Fully Connected Layer

![Fully Connected Layer](../.gitbook/assets/image%20%28174%29.png)

FC Layer에서 하는 일은 어떤 벡터를 가지고 연산을 하는 것이였다. 입력값으로 32 x 32 x 3의 이미지가 있으면, 이 이미지를 길게 펴서 3072차원의 벡터\(3072 x 1\)로 만들었다. 그 벡터와 가중치 W \(10 x 3072\)를 곱해서 Layer의 출력인 activation을 얻는다. \(1 x 10\) 벡터 입력값과 W 를 곱하게 되면 어떤 숫자 하나를 얻게 되는데, 이것에 그 뉴런의 한 값이라고 할 수 있다. 이 예시에서는 10 차원의 데이터를 넣었으니 10개의 output이 나온다고 볼 수 있다.

## Convolution Layer

![Convolution Layer](../.gitbook/assets/image%20%28263%29.png)

Convolution Layer와 기존 FC 레이어의 주된 차이점이 있다면 Convolution Layer는 기존의 구조를 보존시킨다는 것이다. 기존 FC가 입력 이미지를 길게 폈다면 이 레이어는 이미지 구조를 그대로 유지하게 된다. 

5 x 5 x 3 filter가 W가 되는거고, 이 필터로 이미지 위에서 sliding 하면서 공간적으로 내적을 수행하게 된다. 필터는 입력의 depth 만큼 확장된다.\(Filters always extend the full depth of the input volume\) 하나의 필터는 이미지에 비해 작은 영역을 취하지만 \(5 x 5\) depth는 입력값의 depth와 똑같이 취한다. \(3\) 

![Convolution Layer](../.gitbook/assets/image%20%28304%29.png)

이제 이 필터를 이미지의 어떤 공간에 겹쳐놓고 내적을 수행한다. 그리고 필터의 각 W와 이에 해당하는 이미지의 픽셀을 곱해준다. 5 x 5 x 3 + 1 \(bias\) 만큼 연산을 수행하는 거고, 그만큼 파라미터가 존재하는 거다.

Wx 에 관해서 생각해보면, 우선 내적을 수행하기 앞서 W 값을 길게 펼칠 거다. 5 x 5 x 3 형태의 W 를 쭉 펴서 벡터로 만든 형태일 거다. 이해를 위해 이미지로는 위와 같이 표현하곤 하지만, 실제로는 펼친 상태에서 벡터 간 내적을 구하는 것이다.

![Convolution Layer - activation map ](../.gitbook/assets/image%20%2824%29.png)

Convolution은 이미지의 left-top부터 시작해서 필터의 중앙으로 값들을 모으게 된다. 필터의 모든 요소들을 가지고 내적을 수행하면 하나의 값을 얻게 된다. 그 값을 output activation map의 해당 위치에 입력이 되고, 그 이후 다음 영역으로 sliding 한다. 출력 행렬의 크기는 sliding을 어떻게 하느냐에 따라 다르게 되지만 \(stride\) 기본적으로는 한 칸씩 이동하며 연산을 수행한다.

![](../.gitbook/assets/image%20%28225%29.png)

이렇게 하나의 필터를 가지고 전체 이미지에 convolution을 하면서 activation map이라는 출력값을 얻게 된다. 보통 필터마다 다른 특징들을 뽑기 위해 여러개의 필터를 사용한다. 그럼 그 필터 갯수만큼  activation map이 만들어 질 거다.

![](../.gitbook/assets/image%20%28169%29.png)

CNN은 이렇게 Convolution Layer들이 연속된 형태가 되는 거다. 여러 필터를 각각 쌓아 올리게 되면 간단한 Linear Layer 형태의 NN이 된다. 그리고 그 Layer들 사이에 activation function을 넣는다. 그럼 가끔 Pooling Layer도 있을 수도 있지만 일반적으로  ConvLayer - Activation Layer가 반복되는 형태로 만들어 질거다. 각 Layer의 출력은 다음 Layer의 입력이 된다.

각 Layer는 여러개의 필터를 가지고 있고, 각 필터마다 activation map을 만든다. 여러개의 Layer를 쌓고 나면 결국 각 필터들이 계층적으로 학습을 하게 되는 것이다.

Activation map의 depth는 필터의 갯수라고 보면 된다. 어차피 각각의 필터는 내적을 해서 하나의 값만 뽑아내기 때문에 depth는 1일거고, 그 필터들이 겹겹이 되면 쌓이게 되면 그게 바로 depth 값이 된다.

### Features

![feature map](../.gitbook/assets/image%20%2871%29.png)

grid 안의 각 cell들은 하나의 뉴런\(filter\)를 뜻하는 거다. 각 요소는 각 뉴런의 활성을 최대화 시키는 입력의 모양을 나타낸다. 

앞쪽의 필터들은 low level feature들을 학습하게 되고, mid-level 에서는 좀 더 복잡한 특징들을 가지게 된다. high level feature에서는 객체와 좀 더 닮은 것들이 출력으로 나오게 된다. Layer 계층에 따라 단순/복잡한 feature들이 존재하는 것을 알 수 있다. 

지금까지 ConvLayer를 계층적으로 쌓아서 단순한 특징들을 뽑고, 그것을 또 조합해서 더 복잡한 특징들도 뽑게 되는 과정을 봤다. 네트워크 앞쪽은 simple 한 것에 대한 특징을 갖고 뒤쪽으로 갈 수록 특징이 복잡해진다. 강제로 학습시킨게 아니라 계층적으로 쌓아서 backprop 하다보니 자연스럽게 네트워크 스스로 그렇게 만들어진 거다. 

![activations](../.gitbook/assets/image%20%28292%29.png)

Activation의 모양들을 나타낸 이미지다. 각 activation은 이미지가 필터를 통과한 결과값이며, 이미지 중 어느 위치에서 필터가 크게 반응했는지를 보여준다.

![CNN &#xB3D9;&#xC791; &#xC608;&#xC2DC;](../.gitbook/assets/image%20%28173%29.png)

CNN이 동작하는 예시를 보자. 입력 이미지는 여러 Layer를 통과하는 것을 볼 수 있다. 첫번째 ConvLayer 후에는 non-linear\(activation function\)를 통과한다. CNN 끝단에는 FC Layer가 있는데, 이 Layer는 마지막 ConvLayer 출력값 모두와 연결되어 있고, 최종 score를 계산하기 위해 사용한다.

### Filter & Stride

![&#xD569;&#xC131;&#xACF1; &#xACC4;&#xC0B0;&#xC808;&#xCC28;](../.gitbook/assets/image%20%28296%29.png)

Filter는 이미지의 특징을 찾아내기 위한 공용 파라미터이다. Kernel 이라고도 한다. 입력 데이터를 지정된 간격으로 순회하며 채널별로 합성곱을 하고, 그 합을 Feature Map으로 만든다.

![Feature map &#xACFC;&#xC815;](../.gitbook/assets/image%20%28142%29.png)

![](../.gitbook/assets/image%20%28180%29.png)

입력 데이터가 여러 채널을 갖는 경우, 필터는 각 채널을 순회하며 합성곱을 계산한 후, 채널별 feature map을 만든다. 그리고 각 채널의 피처맵을 합산하여 최종 feature map으로 반환한다. 필터 별로 1개의 feature map 이 만들어진다.

![Spatial Dimension](../.gitbook/assets/image%20%28271%29.png)

7 x 7 input에 3 x 3 필터를 취한 것을 예를 들어 보자. 한칸씩 sliding 한다고 했을 때 \(stride = 1\) 좌우 방향으로 5번 수행, 상하 방향으로 5번 수행 가능하다. 5 x 5 의 output이 나오게 되는 거다. stride = 2인 경우는 두칸을 sliding 하여 계산하는 거다. 그 경우는 3 x 3 output이 나온다. 이미지를 sliding 해도 필터가 모든 이미지를 커버 할 수 없는 경우에는 잘 동작하지 않는다. \(ex, stride=3\) 불균형한 결과가 나오기 때문에 권장하지 않는다.

output 사이즈를 수식으로 표현하면 아래와 같이 쓸 수 있다. 

* 입력 사이즈 : N
* 필터\(커널\) 사이즈 : F

$$
output\_size = (N-F)  / stride + 1\\
$$

이를 이용해 어떤 필터 크기를 사용해야 하는 지 알 수 있다. 몇 개의 출력값이 나오는 지, 어떤 stride를 사용했을 때 이미지에 맞는지 등을 알 수 있다.

Stride를 크게 가져갈수록 출력은 점점 작아진다. 이미지를 다운샘플링 하는건 pooling과 비슷하지만 확실히 다르고 더 좋은 성능을 보이기도 한다. 다운 샘플링하는 동시에 더 좋은 성능을 얻을 수 있다. Activation map의 사이즈를 줄이는 것은 모델의 전체 파라미터의 개수에도 영향을 미친다. 파라미터 수, 모델 사이즈, over-fitting 등 trade-off 문제가 있을 수 있다.

### Padding

![](../.gitbook/assets/image%20%28121%29.png)

출력의 사이즈를 원하는 사이즈로 만들기 위해서 zero-padding을 흔히 사용한다. 코너 부분을를 처리할 때도 유용한 방법이다. 이미지의 테두리 부분을 0으로 set 해줌으로써 left-top 위치에서도 필터 연산을 수행할 수 있게 한다. 기존의 output 사이즈 구하는 수식도 이에 따라 아래와 같이 수정된다.

* 입력 사이즈 : N
* 필터\(커널\) 사이즈 : F
* 패딩 사이즈 : P

$$
output\_size = (N + 2P - F) /S + 1
$$

기본 사이즈 N 의 양 끝에 padding\_size 만큼 추가하니까 $$N + (padding\_size)*2$$ 가 되고 그 이후 연산은 동일하다.

Padding을 하게 되면 출력 사이즈를 유지시켜주고 필터의 중앙이 않는 곳도 연산이 가능하다. 

> **Q\) 가로, 세로 stride 사이즈가 항상 동일한가?**  
> A\) input 데이터를 일반적으로 정사각 행렬을 사용하기 때문에, 보통은 모든 영역에 동일한 stride를 사용한다.

![](../.gitbook/assets/image%20%28238%29.png)

보통 필터는 3 x 3 \(padding = 1\) 이나 5 x 5 \(padding = 2\) 사이즈를 일반적으로 사용한다. 

Padding 하지 않고 레이어가 여러 겹 쌓이게 되면 출력 사이즈\(activation map\)는 아주 작아질거다. activation map 이 작다는 건 일부 정보를 잃게 되는 거고 원본 이미지를 표현하기에는 너무 작은 값을 사용하게 되는 거다. 그렇게 줄어드는 이유는 매번 각 코너에 있는 값들을 계산하지 못하기 때문에 그 부분에 대한 데이터를 잃는 거다.

![](../.gitbook/assets/image%20%28275%29.png)

입력이 32 x 32 x 3 일 때 5 x 5 필터가 10개 있고 stride = 1, padding = 2 일 때 이 레이어의 파라미터는 몇 개일까?

각각의 5 x 5 x 3 필터에는 하나의 bias term이 있다. 각 필터당 5 \* 5 \* 3 + 1 = 76 개의 파라미터가 있는거고 이 필터가 10개 있으니 전체 760개의 파라미터가 존재한다.

![](../.gitbook/assets/image%20%28300%29.png)

필터 사이즈, stride, padding 의 일반적인 설정값

* 필터 크기 : 3 x 3, 5 x 5
* 필터의 갯수 : 2의 제곱수 \(32, 64, 128 .. \)
* stride : 1 \(3 x 3 일 땐 보통 1\) or 2
* padding : 그 설정에 따라 조금씩 다름. 보통 1 or 2

### 1 x 1 Convolution

![](../.gitbook/assets/image%20%28332%29.png)

 1 x 1 convolution 도 의미가 있다. 공간적인 정보를 이용하지는 않지만 여전히 depth 만큼 \(1 x 1 x depth\)  연산을 수행한다. 전체 dept에 대한 내적을 수행하는 거다. 56 x 56 x 64 에 1 x 1 convolution을 수행하면 output은 56 x 56 x 32 이다.

### Neuron view

![](../.gitbook/assets/image%20%28116%29.png)

ConvLayer를 뉴런에 비유해보자. 입력이 들어오면 W\(필터값\)와 곱하고 하나의 값을 출력하는 구조가 유사하다. 차이점은 뉴런은 Local connectivity가 있다는 거다. ConvLayer처럼 슬라이딩 하는게 아니라 특정 부분에 연결되어 있다.  하나의 뉴런은 한 부분만 처리하고 그런 뉴런들이 모여 전체 이미지를 처리하는데, 이러한 방식으로 spatial structure를 유지한채로 layer 출력인 activation map을 만드는 거다.

![](../.gitbook/assets/image%20%2862%29.png)

필터 하나는 뉴런의 receptive field 라고 표현할 수 있다. 이는 한 뉴런이 한번에 수용할 수 있는 영역을 의미한다. 필터의 값은 항상 동일하므로 레이어 간 파라미터들을 공유한다.\(Parameter Sharing\)

![](../.gitbook/assets/image%20%28108%29.png)

오른쪽 박스는 5개의 필터를 사용해 만들어진 output 이다. 3D grid로 표현해 볼 수 있다. \(28 x 28 x 5\)

Depth 평면으로 바라봤을 때 위의 5개의 점은 같은 영역에서 추출된 서로 다른 특징이다. 각 필터는 서로 다른 특징을 추출하는데, 이미지의서 같은 영역에서 몇 개의 필터를 통해 다양한 특징들을 추출한다.

FC Layer는 32 x 32 x 3을 모두 펼쳐 1차원 vector로 만든 다음에\(전체를 concat 한다\) 계산을 하고  ConvLayer는 지역정보만 이용해 계산을 한다는 점이 다르다.

## Pooling Layer

![](../.gitbook/assets/image%20%28151%29.png)

### Role

* Pooling Layer는 representation들을 더 작고 관리하기 쉽게 해준다. 
  * Representation은 왜 작게 만드나?
    * 작아지면 파라미터의 수가 줄기 때문. 
    * 일종의 공간적인 invariance\(불변성\)을 얻을 수도 있다. 
* Down-sampling 한다.
  * Input size를 공간적으로 줄여준다. Depth에는 영향을 주지 않는다. 

![](../.gitbook/assets/image%20%28111%29.png)

일반적으로 max pooling을 사용한다. 단순 연산이기 때문에 학습 파라미터가 없다.

### Max Pooling

![](../.gitbook/assets/image%20%28307%29.png)

Pooling Layer에도 필터 크기를 정할 수 있다. 얼마만큼의 영역을 한번에 묶을지 정하는 거다. 위의 예시는 2 x 2필터와 2 stride 사용한다. 

Pooling Layer도 ConvLayer처럼 sliding 하며 연산한다. 대신 내적을 하는게 아니라 단순히 해당 영역에서 가장 큰 값을 뽑아낸다. 

> **Q\) Max pooling이 Avg pooling보다 더 좋은 이유는?**   
> **A\)** 우리가 다루는 값들은, 얼머나 이 뉴런이 활성화 되었는지를 나타내는 값들이다. 즉, 이 필터가 각 위치에서 얼마나 활성화 되었는지를 값으로 표현한 것이다. Max pooling은 그 지역이 어디든, 어떤 신호에 대해 얼마나 그 필터가 활성화되었는지를 알려주는 것이다. 그 값이 어디에 있는지보다는 그 값이 얼마나 큰지가 중요하다는 것이 Max pooling의 직관이다.

> **Q\) Pooling과 stride의 차이는?**  
> **A\)** 많은 사람들이 down sample할 때 pooling보다 stride를 많이 사용하는 추세이다. Pooling도 일종의 stride 기법이라고 볼 수 있다. 요즘은 stride가 더 좋은 성능을 보이는 것 같다. pooling 대신 stride를 써도 된다.

### Design Choice

![](../.gitbook/assets/image%20%28308%29.png)

W\(width\) x H\(height\) x D\(depth\) 에 F\(필터 사이즈\), S\(stride\)를 추가하여 pooling layer를 디자인할 수 있다.

* 입력 데이터 폭 : W
* 입력 데이터 높이 : H
* 입력 데이터 채널 : D

$$
OutputWidth= (W - F)/S + 1 \\
OutputHeight = (H - F)/S + 1 \\
OutputDepth = D
$$

보통 pooling layer에서 padding을 하지 않는다. 어차피 Down sampling 이 목적이고, 코너의 값을 계산하지 못하는 경우도 없기 때문이다. 

가장 널리 쓰이는 필터 사이즈는 2 x 2, 3 x 3이고 stride는 보통 2를 사용한다.

## Layer Patterns

![CNN &#xB3D9;&#xC791; &#xC608;&#xC2DC;](../.gitbook/assets/image%20%28173%29.png)

마지막 ConvLayer 출력은 3D volume으로 이루어질거고\(W x H x D\), 마지막 레이어에서는 이 값들을 전부 stretch 하여 1차원 vector로 만들어서 FC Layer 입력으로 사용한다. 그럼 Convolution Network의 모든 출력을 서로 연결하게 되는거다. 

마지막 Layer\(FC Layer\)부터는 spatial structure를 신경쓰지 않는다. 전부 다 하나로 통합시키고 최종적인 추론을 하는 거다. 그렇게 되면 scores가 출력으로 나온다. 

input \(3-D volume\) -&gt; **ConvLayer** -&gt; activation map -&gt; **Activation function**  
input \(3-D volume\) &gt; **Pooling Layer** -&gt; down sampling  
input \(1-D vector\) -&gt; **FC Layer** -&gt; scores \(1-D vector\)

일반적으로 ConvNet의 패턴을 아래와 같이 표현할 수 있다.

 `INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`

> **Example\)**
>
> * INPUT -&gt; FC : Linear classifier
> * INPUT -&gt; CONV -&gt; RELU -&gt; FC
> * INPUT -&gt; \[CONV -&gt; RELU -&gt; POOL\] \* 2 -&gt; FC -&gt; RELU -&gt; FC
> * INPUT -&gt; \[CONV -&gt; RELU -&gt; CONV -&gt; RELU -&gt; POOL\] \* 3 -&gt; \[FC -&gt; RELU\] \* 2 -&gt; FC



{% hint style="info" %}
[http://cs231n.github.io/convolutional-networks/](http://cs231n.github.io/convolutional-networks/)
{% endhint %}





