# 9. CNN Architectures

![](../.gitbook/assets/image%20%28216%29.png)

![](file:///C:\Users\genius\AppData\Local\Temp\msohtmlclip1\01\clip_image004.jpg)

1998 LeNet

산업에 아주 서옥ㅇ적으로 적용된 최초의 ConvNet

Stride=1 5x5 필터 거치고 마지막에 FC 레이어. 숫자 인식이 엄청난 성공을 거둠

![](../.gitbook/assets/image%20%2855%29.png)

AlexNet

Large scale CNN

ImageNEt classification task 잘 수행

ConvNEt 연구의 부흥을 일으킴

Conv-pool-norm 구조가 두번 반복. Conv layer가 봍고 마지막에 fc layer가 몇 개 붙음

기존의 lenet과 유사하지만 레이어가 많아짐.

AlexNet 모델 사이즈.

Input : 227 227 9

Conv Layer 11x11 필터, stride=4가 96개

Pooling layer. 3x3 Stride=2,

ReLU 사용.

Local response normalization layer: 채널간의 normalization 을 위한건데 요즘은 사용하지 않는다. 큰 효과가 없어서

Data augmentation : flipping, jittering, color norm 등을 적용

Dropout

batch size 128

Sgd momentum

L\_r= 1ed-2. 학습이 종료되는 시점까지 1e-10까지 줄임

Weight decasy 사용

모델 앙상블

ConvNet과 유사한데 한가지 차이가 있다. 모델이 두개로 나눠져서 서로 교차하는 거다. 전체 레이어를 GPU에 넣을 수가 없어서 네트워크를 gpu에 분산시켜서 넣었따. 각 gpu가 모델의 뉴런과 feature map을 반반 가지고 있따.

Conv 1, 2, 4, 5에서는 gpu 내에 있는 feature map만 사용한다. 전체 96개의 feature map을 볼 수 없었다. 48개의 feature map만 이용했따.

Conv3, fc 6, 7, 8은 전체 feature map과 연결되어 있따. Gpu 간의 통신을 하기 때문에 이전 입력 레이어의 전체 feature map을 받아올 수 있었다.

Cnn 아키텍쳐의 베이스 모델로 사용했다. 아직도 꽤 사용한다. Transfer learning에 많이 사용되었다.

![](../.gitbook/assets/image%20%28239%29.png)

2013년 imagenet challange승자는 zfnet이라는 모델. 대부분 alexnet 하이퍼파라미터 수정함

레이어 수와 구조는 같지만 stride, filter 수 같은 하이퍼파라미터를 조정하여 alexnet의 error rate를 개선했다.

![](../.gitbook/assets/image%20%28108%29.png)

2014년. 네트워크가 훨씬 더 깊어졌다. 2012/13에는 8개였는데 19, 22레이어로 늘어났다.

GoogleNet, VGGNet이 각각 1, 2위 차지했다.

![](../.gitbook/assets/image%20%28128%29.png)

VGGNet

훨씬 더 깊어졌고 더 작은 필터를 사용했따. 16~19 레이어를 가진다. 3x3필터만 사용한다. 가장 작은 필터이다. 작은 필터를 유지해주고 주기적으로 pooling을 수행. 심플하면서도 고급진 아키텍쳐.

필터의 크기가 작으면 파라미터의 수가 더 적다. 큰 필터에 비해 레이어를 좀 더 많이 쌓을 수 있다. Depth를 더 키울 수 있다는 거다. 3x3 필터를 여러 개 쌓은 것은 결국 7x7 필터를 사용하는 것과 실질적으로 동일한 receptive filter를 가지는 것이다.

Stride=1, 3x3 필터가 3개 있으면 receptive filter는? Effective receptive fields는 7x7 이다. 하나의 7x7 필터를 사용하는 것과 동일하다. 7x7 사용하는 것과 같은 효과를 가지면서 더 깊은 레이어를 쌓을 수 있게 된다. Non-linearity를 더 추가할 수도 있고 파라미터 수도 더 적어지게 된다.

![](../.gitbook/assets/image%20%28278%29.png)

![](../.gitbook/assets/image%20%28292%29.png)

Vgg16 레이어ㅇ 수는 16개. Vgg19는 convllayer가 조금 더 추가되어서 19개

Forward pass시 필요한 전체 메모리는 100MB이다. Backward pass가ㅗ려한면 더 많으 메모리가 필요하다 .

Vgg16는 메모리 사용량이 많은 편이다. 전체 파라미터 개수는 138M 개. Alexnet은 60M

Fc7은 4096 사이즈의 레이어인데 아주 좋은 feature representation 을 가지고 있는 것으로 알려졌다. 다른 데이터에서도 feature 추출이 잘 되며 다른 task에서도 일반화 능력이 뛰어나다. 

![](../.gitbook/assets/image%20%28159%29.png)

Googlenet도 깊다. 22개의 레이어 가지고 있다. 높은 계산량을 효율적으로 수행하도록 네트워크를 디자인했다. Inception module을 여러 개 쌓아서 만든다. 파라미터 수를 줄이기 위해 FC Layer를 사용하지 않았다. 전체 파라미터 수가 5M 정도로, alexNet 보다 적다.

Inception module : good local network typology를 구성하고 싶었따. Local topology를 구현했고 이를 쌓아올렸다.

Inception module 내부에는 동일한 입력을 받는 서로 다ㅇ른 필터들이 병렬로 존재한다. 다양한 conv 연산을 병렬적으로 수행하는 거다. 각각의 출력값들이 나오는데 그 출력을 하나의 depth로 concat 시킨다. 하나의 tensor로 출력이 결정된다.

계산비용\(computational complexity\)에 문제가 있다.

![](../.gitbook/assets/image%20%28226%29.png)

각각의 출력값들을 concat 한 사이즈를 보자. 28x28 은 동일하고 depth가 쌓이게 된다. 28x 28x672가 된다.

입력은 28x28x256 이고 출력은 28x28x 672가 됐다. Spatial dimension은 변하지 않았지만 depth가 엄청 늘어났다. Spatial dimension을 맞추기 위해 zero padding 했다.

![](../.gitbook/assets/image%20%2812%29.png)

각 필터의 연산양은 다음과 같다. 연산량이 아주 많다. Pooling layer도 입력의 depth를 그대로 유지하고 여기에 다른 레이어의 출력이 계속 더해지기 때문에 레이어를 거칠수록 depth가 많이 늘어난다.

Bottleneck layer를 사용. Conv 연산 수행하기 전에 입력이 더 낮은 차원으로 보내는 거다.

![](../.gitbook/assets/image%20%28288%29.png)

1x1  jconv는 각 spatial location 에서만 내적을 수행한다. 그러면서 엗소만 줄일 수 있다. 입력의 depth 를 더 낮은 차원으로 projection 시키는거다. Input feature map 들 간의 선형 결합\(linear combination\)이라고 할 수 있다.

입력 레이어의 depth를 줄이는 거다. 1x1 conv를 통해.

![](../.gitbook/assets/image%20%28230%29.png)

3x3, 5x5 conv 이전에 1x1 conv를 추가하고, pooling layer 후에도 1x1 conv를 추가한다. 1x1 conv가 bottleneck layer로 추가되는거다.

![](../.gitbook/assets/image%20%28109%29.png)

Bottleneck 레이어를 추가하고 나서 computation 이 많이 줄어들었다. \(1x1 conv가 depth의 차원을 줄여주니까\) Conv 입력 depth가 줄어들었다. 1x1 conv를 이용하면 계산량을 조절할 수 있다. 정보 손실이 발생할 수는 있지만, 동시에 redundancy가 있는 input features를 선형 결합 한다고 볼 수 있다. 1x1 conv로 선형결합을 하고 non-linearity를 추가하면\(relu 같은\) 네트워크가 더 깊어지는 효과도 있다. 일반적으로 도움이 되고 더 잘 동작한다.

![](../.gitbook/assets/image%20%28248%29.png)

초기 6개 레이어는 일반적인 conv-pool-conv-pool의 일반적인 네트워크 형태이다. 이후에는 inception moudle에 쌓이는데 모두 조금씩 다르다. 그리고 마지막이 classifer 레이어를 추가한다. 계싼량이 많은 fc layer를 대부분 걷어냈고 파라미터가 줄어들어도 성능은 유지 시켰다.

중간의 추가적인 줄기는 보조 분류기\(auxiliary classifier\) 이다. 평범한 미니 네트워크다. \(avg pooling, conv, fc layer, softmax\) 이 곳에서도 trainset loss를 계산한다. 네트워크가 깊기 때문에 이 중간 두곳에서도 loss를 계산한다. 보조 뷴류기를 중간 레이어에 달아주면 추가적인 gradient를 얻을 수 있고 중간 레이어의 학습을 도울 수 있다.

정리하면 다음과 같다.

-       총 22개의 레이어를 가지고 있다

-       각 inception module은 1x1/3x3/5x5 conv layer를 병렬적으로 가지고 있다.

-       Fc layer를 들어냈다

-       alexnet보다 12배 작은 파라미터

2015년 우승자 REsNet

혁명적으로 네트워크 깊이가 깊어짐. 152개!

Residual connections .

![](../.gitbook/assets/image%20%28222%29.png)

깊은 네트워크가 항상 좋은 건 아니다. 더 깊은 네트워크가 안좋을 수도 있다. Overfit?은 아니다. Training error도 낮다. 더 깊은 모델 학습 시 optimization 에 문제가 생긴거라고 생각함. 최적화가 어렵다고 판단했다.

더 얕은 모델의 가중치를 깊은 모델의 일부 레이어에 복사하고 나머지 레이어는 identity mapping\(input을 ouput으로 그냥 보냄\)

![](../.gitbook/assets/image%20%2890%29.png)

이를 네트워크에서 구형하기 위해 레이어를 단순히 쌓기만 하지 않았따. Direct mapping 대신 Residual mapping 하도록 레이어를 쌓는다.

레이어가 직접 H\(x\)를 학습하기 보다, H\(x\) – x를 학습할 수 있도록 만들어준다.  이를 위해 skip connection을 도입하게 된다. 가중치가 없으면 입력을 identity mapping으로 그대로 출력단으로 내보낸다. 그러면 레이어는 delta 만큼만 학습하게 된다. 입력 X에 대한 residual이라고 할 수 있다. 최종 출력값은 input X + residual\(변화량\) 이다.

네트워크는 residual 만 학습하면 되는 거다. 출력 값도 결국 입력 X에 가까운 값이다. 레이어가 full mapping을 학습하기 보다 이런 약간의 변화만 학습하는 거다.

깊은 레이어에서는 H\(X\)를 배우는게 힘들다. 그래서 H\(x\)를 직접 배우는 대신에 X에 얼마의 값을 더하고 빼야 하는지 배우는 것\( residual\)이 쉬울 것이라고 생각한거다. 입력값을 어떻게 수정할지를 배우는 거다.

![](../.gitbook/assets/image%20%28162%29.png)

하나의 residual blocks은 두 개의 3x3 conv layer로 이루어져 있다. 이렇게 구성해야 잘 동작한다. 이 residual 을 깊게 쌓아올린다. Resnet은 150레이어까지 쌓았다. 주기적으로 필터를 두배씩 늘리고 stride2를 통해 downsampling 한다. 초반에는 convLayer가 붙고 끝에는 FC 레이어 대신에 global average pooling layer를 사용한다. 하나의 map 전체를 average pooling 하는 거다. 그리고 마지막엔 클래스 분류를 위한 노드가 붙는다.

![](../.gitbook/assets/image%20%28174%29.png)

Resnet의 경우 모델 depth가 50 이상일 때 bottleneck layer를 도입한다. 1x1 conv를 도입해서 초기 필터의 depth를 줄여준다.

-       모든 convlayer 다음 batch norm 사용

-       초기화는 Xavier \(scaling factor추가\)

-       SGD+Momentum

-       Learning rate은 스케쥴링을 통해 validation error 가 줄어들지 않는 시점에서 조금씩 줄여준다.

-       Mini batch 256

-       Weight decay

-       Dropout 사용 x

resnet에서는 네트워크가 깊어질수록 training error는 더 줄어들었음. 레이어와 깊다고 Train error가 더 높아지는 경우는 없었다. ILSVRC, COCO에서 2위와 엄청난 격차로 1위 했다. “인간의 성능”보다 뛰어난 수치를 넘어섰따 \(5% error\)

최근에 사람들이 많이 사용하는 네트워크다.

![](../.gitbook/assets/image%20%28170%29.png)

Complexity 그래프

조금 변형된 모델들도 있다. Google-inception 모델은 버전 별로 v2, v3 등이 있는데 가장 좋은 모델은 Inception-v4 : Resnet + Inception 모델이다.

원 : 메모리 사용량

X : 연산량

Y : accuracy

VGGNet : 효율성이 작다. 메모리도 엄청 잡아먹고 계산량도 많다. 뭐 성능은 나쁘지 않다.

GoogleNEt : 가장 효율적인 네트워크. 메모리 사용도 적다.

AlexNet : accuracy 낮고 계산량도 작다. 메모리 사용량은 비효율적

Resnet : 적당한 효율성. 메모리사용량 계싼량이 중간정도, accuracy는 최상위



최근 연구분야나 역사적으로 의미 있는 아키텍쳐 소개 고고

![](../.gitbook/assets/image%20%28224%29.png)

2014 network in network

-       MLP conv layer. 네트워크 안에 작은 네트워크를 삽입

-       각 Conv layer 안에 MLP를 쌓는다.

-       처음에는 기존의 Conv Layer가 있고 FC Layer를 통해 abstract features 를 잘 뽑을 수 있도록 한다. 잔순히 ocnv filter만 사용하지 말고 더 복잡한 계층ㅇ을 만들어 activation map으 얻어보고자 하는 아이디어.

-       기본적으로 FC layer\(1x1 conv layer\) 사용한다.

-       Googlenet, resnet 보다 먼저 bottleneck 개념을 정립했다.

Resnet 성능을 향상시킨 최근의 연구들.

![](../.gitbook/assets/image%20%28168%29.png)

2016. resnet 블록 디자인을 향상시킨 논문.

이 논문에서는 Resnet block path를 조절했다.

Direct path를 늘려 정보들이 앞으로 더 잘 전달되고 backprop도 더 잘 될 수 있게 개선

![](../.gitbook/assets/image%20%2847%29.png)

2016

기존의 resnet 논문의 깊이 쌓는 거에 집중했지만 이 논문은 Depth보다 residual 이라고 주장. Residual block을 더 넓게 만들었다. \(conv layer 필터를 더 많이 추가\) 각 레이어를 넓게 구성했더니 50레이어만 있어도 152개의 레이어보다 성능이 좋다는 걸 확인. 네트워크의 depth 대신 filter의 width를 늘리면 계산 효율도 증가. \(병렬화가 더 잘되기 때문\) 네트워크의 depth를 늘리는 것은 sequential한 증가이기 때문에 conv 필터를 늘리는 편이 더 효율적이다.

![](../.gitbook/assets/image%20%28289%29.png)

ResNeXt \(resnet + inEXeption\)

비슷한 시점에 또 등장한 논문

Residual block의 width에 집중. Filter의 수를 늘리는 것. Residual block 내에 다중 병렬 경로를 추가.  Thinner blocks을 병렬로 여러 개 묶었다. 위의 wide resnet도 그렇고 여러 layer를 병렬로 묶어준다는 점에서 inception module과도 연관이 있다.

![](../.gitbook/assets/image%20%2882%29.png)

네트워크가 깊어질수록 vanishing gradient문제가 발생하는데,

Train time에 레이어의 일부를 제거한다. Short network 면 트레이닝이 더 잘 될 수 있읜까

일부 네트워크를 골라 identity connection으로 만든다 \(weight를 0으로 설정\)

그렇게 shorter network 를 만들면 train할 때 gradient가 더 잘 전달 될 수 있다.

Droptout과 유사함.

Test time에는 full network 를 사용

Resnet과 견줄만한 성능의 모델들

fractalNet.

Residual connection이 필요없다고 주장함. Shallow/deep network 정보 모두를 잘 전달하는게 중요하다고 생각해서 모두 output에 연결했다. Train time 에는 ropout처럼 일부 경로만을 이용해서 train, test time에는 full network 사용. 좋은 성능을 입증함

![](../.gitbook/assets/image%20%28240%29.png)

DenseNet

Dense block. 한 레이어가 그 레이어 하위의 모든 레이어와 연결이 되어 있다. Network 입력이 모든 layer의 입력으로 들어간다. 모든 레이어의 출력이 각 레이어의 출력과 concat 된다. 이 값이 각 convlaye입력으로 들어간다. Dimention 줄여주는 과정 포함. Vanishing gradient 문제를 완화할 수 있다고 주장. Feature를 더 잘 전달하고 더 잘 사용할 수 있게 해준다. 각 레이어의 출력이 다른 ㄹ에이어에서도 여러 번 사용될 수 있기 때문.

![](../.gitbook/assets/image%20%2831%29.png)

레이어 간의 연결을 어떻게 할 지, depth를 어떻게 구성할 지에 대한 연구가 많다

Googlenet은 표율적인 모델에 대한 방향을 제시 했었는데 squeezeNEt도 그 효율성을 강조한다.

squeezeNet

fire module를 도입해

squeeze layer는 1x1 filter로 구성되고, 이 출력 값이 1x1/3x3 필터들로 구성되는 expand layer의 입력이 된다.

Squeezenet은 alexnet 만큼의 accuracy지만 파라미터는 50배 더 작다. 용량의 0.5m밖에 안됨

