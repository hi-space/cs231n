# 7. Training Neural Network 2

  
Weight Initialization

-       Initialization too small : Activation 이 점점 0에 가까워 지고, gradient도 0이 되어서 학습이 일어나지 않는다

-       Initialization too big : Activation이 saturate 되어 gradient가 0이 되고, 학습이 일어나지 않는다.

l  Fancier optimization

l  Regularization

l  Transfer Learning

Sgd 문제점

![](../.gitbook/assets/image%20%28260%29.png)

이런 함수에서 gradient 방향이 고르지 않다. 고차원 공간에서 빈번하게 발생하는 문제이다.

![](../.gitbook/assets/image%20%28160%29.png)

Local minima, saddle point.

X축은 하나의 가중치, Y축은 loss

Gradient가 0이 되기 때문이 locally flat 해서 SGD가 멈춘다. SGD는 gradient가 0인 지점을 만나면 다른 방향으로움직이게 되는데, opposite gradient도 0이라서 멈춘다.

Saddle point에서도 gradient=0 이라서 멈춘다.

Local minmima 보다 saddle point에 취약하다. Saddle point 근처에서도 문제가 생긴다. Gradient가 0은 아니지만 기울기가 아주 작다. 업데이트가 아주 느려진다.

매번 모든 loss를 구할 수 없어 mini batch로 나눠서 loss를 구하게 되기 때문에 gradient의 정확한 값을 얻는게아니라 추정값\(noisy estimate\)를 구하는 것이다. Minima 까지 가는데 오랜 시간이 걸린다.

![](../.gitbook/assets/image%20%2831%29.png)

Gradient를 계산할 때 velocity를 이용하는 거다. 미니배치의 gradient 방향만 고려하는게 아니라 velocity를 같이 고려한다. 하이퍼파라미터 rho\(memoemtum 비율\) 보통 0.9로 맞춤. Velocity에 일정 비율을 곱하고 현재 gradient를 더한다. Gradiertn vector그대로 방향이 아닌 velocity grradien 바얗ㅇ으로 간다. 간단하지만 앞선 문제들을 해결 할 수 있따.

![](../.gitbook/assets/image%20%28208%29.png)

Local minima에 도달해도 velocity를 가지고 이씩 때문에 graidnerr=0이라도 움직일 수 있따. Local minima를 극복하게 된다. Saddle point도 주변의 gradient가 작더라도 속도가 있기 때문에 saddle point를 극복하고 계쏙 밑으로 내려올 수 있따. Momentum을 추가하면 velocity가 생기면서 결국 noise가 평균화된다. 보통의 SGD가 구불구불 움직이는 것에 비해 momemum은 minima를 향해 더 부드럽게 움직인다

![](../.gitbook/assets/image%20%2848%29.png)

Momentum 변형. 계싼하는 순서를 조금 바꿨따. 기본 SGD momentum은 현재 지점에서의 gradient를 계산한 뒤에 velocity와 섞어준다. Nesterov momentum은 일단 velocity 방향으로 옴직여서 그 지점에서의 graidnet를 계산한다. 그리고 다시 원점에 돌아가 둘을 합친다. Velocity의 방향이 잘못되었을 경우에 현재 gradient 방향을 좀 더 활용할 수 있도록 해준다. convex opticminzatoni 에서는 잘 동작하지만 NN 같은 non-convex optimization에서는 성능이 보장되지는 않는다.

직관적으로 보면 velocity는 이전 gradient의 weighted sum 이다.

Adagrad

훈련 도중 계산되는 gradients를 활용하는 방법이다. Velociry term 대신 grad squared sum을 이용한다. Gradient에 제곱을 해서 계쏙 더한다. Update term을 앞거 셰산한 gradient 제곱항으로 나눠준다.

![](../.gitbook/assets/image%20%28321%29.png)

학습이 계속 진행되면 학습횟수 t가 계속 늘어나면? Step을 진행할수록 값이 작아진다. Update 동안 gradient의 제곱이 계속 더해지기 때문에 estimate 값은 서서히 증가하게 된다. Step size를 더 작게 만든다. convex에서는 m,inima에 근접하며 서서히 속도를 줄여 수렴할 수 있게하겠찌만, non-convex에서는 saddle point에 걸렸을 때 멈춰버릴 수 있다.

![](../.gitbook/assets/image%20%28207%29.png)

이를 변형한 RMSProp

AdaGrad의 gradient 제곱항을 그대로 사용하지만, 이전처럼 그저 누적만 시키는 것이 아니라 기존의 누적 값에 decay\_rate를 곱해준다. Graidnets의 제곱을 계속해서 누적해 나간다. Decay\_rate은 보통 0.9, 0.99 자주 사용. AdaGRAD처럼 gradient 제곱을 계속 나눠주기 때문에 Step 의 속도를 가감속 시킬 수 있다.

RMSProp은 각 차원마다의 상황에 맞도록 적절하게 trajectory를 수정시킨다. AdaGRAd는 잘 쓰지 않는다. Adagrad learning rate늘리면 rmsprop와 비슷하게 움직이겠찌만, 실제로는 잘 사용하지 않는다.

![](../.gitbook/assets/image%20%28358%29.png)

ADAM

Adam은 first moment와 second moment를 이용해 이전의 정보\(Estiamte\)d를 유지시킨다. First moment\( gradient의 가중합\) second monment\(adaggrad, rmsp처럼 gradit제곱 이용\) update를 진행하게 되면 first moment는 velocity를 담당하고 second\_moment는 gradient의 제곱항이다. 둘 다 이용하지만 문제가 있따. 초기 step 에서는 second moment를 0으로 초기화 한다. 업데이트 한 후에도 second dmoment는 여전히 0에 가깝다. 나눠주는 값이 크기 때문에 ㅊ초기 step이 아주 크다.  

ADMA은 초기 step이 엄청 커져버릴 수 있고 그로 인해 잘못될 수도 있따. 이를 해결하기 위해 bias correction term 을 추가한다.

![](../.gitbook/assets/image%20%28120%29.png)

First/second moments를 update 하고 난 후에 현재 step에 맞는 적절한 unbiased term을 계산해준다. 이렇게 만들어진 Adam은 많은 문제에서 잘 동작한다.

문제점도 물론 있다. Adam 이용하면 각 차원마다 적절하게 속도를 높이고 줄이면서 독립적윽로 step을 조절할 거다. 차원의 해당하는 축만을 조절할 수 있기 때문에 차원을 회전 시킬 수 없다.

![](../.gitbook/assets/image%20%28235%29.png)

Learning rates decay 전략이 있다. 각각의 learning rates의 특성을 적절히 이용하는 거다. 처음에는 learning rates를 ㅌ높게 하고 학습이 진행될수록 점점 낮추는 거다.  e학습 과정 동안에 꾸준히 learning rate를 감소시킬 수도 있다.

![](../.gitbook/assets/image%20%28249%29.png)

ResNEt 논문에 나오는 거. Step decay learning rate 전략을 사용한거다. 평평하다가 갑자기 내려가는 구간은 learning rate을 낮춘 구간이다. 현재 수렴을 잘 하고 있는 상황에서 gradient가 점점 작아지고 있으면 learning rate이 너무 높아서 더 깊게 들어가지 못한다. 이 때 learning rate을 낮추면 속도가 줄어들고 지속해서 loss가 내려갈 수 있을 거다.

Learning rate decay는 adam 보다 SGD momentuㅡ 사용할 때 자주 쓴다.

일반적으로 learning rate decay를 학습 초기부터 고려하지 않는다. 처음엔 없다 생각하고 learning rate을 잘 선택하는게 중요하다. Loss curve를 보다가 decay가 필요한 구간에서 decay 해준다.

![](../.gitbook/assets/image%20%28107%29.png)

이 gradient 정보를 이용해 손실함수를 선형 함수로 근사시킨다. 1차 근사함수를 loss함수라고 가정하고 step을 내려간다. 현재 사용하는 정보는 1차 미분값일 뿐이다. 2차 근사 정보를 추가적으로 활용하는 방법이 있따.

![](../.gitbook/assets/image%20%28245%29.png)

2차 근사를 이용하면 minima에 더 잘 근접할 수 있다. 2차 근사 함수를 만들어서 이 2차 근사 함수의 minima로 이동한다.

![](../.gitbook/assets/image%20%2841%29.png)

다른 optimization 함수들과 다른 건, leraning rate이 없다는 점이다. 기본 newton step에는 learning rate이 없지만, 우린 여전히 learning rate이 필요하다. Minima로 이동하는 게 아니라 minima의 방향으로 이동하기 때문이다. 근데 depp elarn 에서는 못쓴다. N \(파라미터 수\) x N 행렬을 계산할 메모리가 없다. 그래서 Full hessian을 그대로 사용하기보다는  Quasi-newton 을 사용한다. Low rank- approximation 하는 방법이다.

![](../.gitbook/assets/image%20%28105%29.png)

L-BFGS도 second-order optimizer 이다. 이것도 hassian을 근사시켜서 사용하는 방법이다. 하지만 2차 근사가 stochastic case에서 잘 동작하지 않기 때문에 DNN에서는 잘 쓰지 않는다. Non-convex problem에서도 적합하지 않다.

![](../.gitbook/assets/image%20%2861%29.png)

실제로는 ADAM 제일 많이 쓰지만 full batch update가 가능하고 stochasticity이 적은 경우라면 L-BFGS가 좋은 선택일 수 있다. \(EX, style transfer\)

![](../.gitbook/assets/image%20%28221%29.png)

Optimization 알고리즘들은 training error를 줄이고 손실함수를 최소화시키기 위한 역할을 수행한다. 하지만 중요한건 test error와의 격차를 줄이는ㄱ거다.

앙상블.

1.     Train multiple independent models

2.     At test time average their results

모델을 하나만 학습시키지 않고 10개의 모델을 독립적으로 학습시키는 거다. 결과는 10개 모델 결과의 평균을 이용한다. 모델의 수가 늘어날수록 overfitting이 줄고 성능이 향상된다. 보통 2%정도 증가.

모델 사이즈, learning rate, 다양한 regularization 기법 등을 앙상블 할 수 있다.

![](../.gitbook/assets/image%20%28104%29.png)

-       모델을 독립적으로 학습심키지 않고 학습 도중 중간 모델들을 저장\(snapshots\)하고 앙상블로 사용할 수 있다. Test time 에는 여러 snopshot에서 나온 값을 평균내서 사용한다.

-       Learning rate을 엄청 낮췄따가 높혔다가 반복하면서 loss함수의 다양한 지역에 수렴할 수 있도록 만들어준다. 그리고 이런 앙상블 기법으로 모델을 한번만 train 시켜도 좋은 성능을 얻을 수 있게 하는 방법이다.

![](../.gitbook/assets/image%20%28360%29.png)

-       Polyak averaging

-       학습하는 동안에 파라미터의 exponentially decaying average를 계속 계산해서 학습 중인 네트워크의 smooth ensamble 효과를 낸다. checkpoints에서의 파라미터를 그대로 사용하지 않고 smoothly decaying average를 사용하는 방법이다. 시도해볼만 하지만 common 한건 아니다

단일 모델의 성능을 향상시키기 위해서는? 앙상블 하려면 test time에 10개의 모델을 돌려야 할 수도 있따. 단일 모델의 성능을 올리는 게 중요하다.

모델이 taringing set에 fitㅎ ㅏ는걸 방지한다.

Loss에 추가 항을 삽입하는 방법이 있었따. L2 regularizioan 은 NN에는잘 어울리지 않는다.

![](../.gitbook/assets/image%20%28132%29.png)

nn에서 가장 많이 사용하는 regularizioatn은 dropout이다. Forward pass과정에서 일부 뉴런을 0으로 만드는 거다. Forward pass 할 때 마다 0이 되는 뉴런이 바뀐다. Dropout은 한 레이어씩 진행하게 된다. 한 레이어의 출력을 전부 구하고 임의의 일부를 0으로 만든 후에 다음 레이어로 넘어가는 식이다. Forward pass iteration 마다 그 모양은 계속 바뀔거다.

Dopout은 fc layer에서 흔히 사용되지만 convlayer에서도 종종 볼 수 있다. convNEt의 경우에는 전체 feature map에서 dropout을 시행한다.

![](../.gitbook/assets/image%20%28106%29.png)

![](../.gitbook/assets/image%20%2833%29.png)

왜 dropout이 좋을까? 일부 값들을 0으로 만들며 training time의 네트워크를 훼손시키는데. Feature 간의 co-adaptation을 방지하는 거다. Dropout을 적용하면 네트워크가 어떤 일부 feature에만 의존하지 못하게 해준다. 다양한 feature를 골고로 이용하게 한다.

단일 모델로 앙상블 효과를 가질 수 있다. 뉴런의 일부만 사용하는서브네트워크인데, droout으로 만들 수 있는 서브네트워크 경우의 수가 아주 많다. Dopout은 서로 파라미터를 공유하는 서브네트워크 앙상블을 동시에 학습시키는 거라고 생각할 수 있다. 거대한 앙상블 모델을 동시에 학습시키는 거라고 볼 수 있다.

Test time 에 dropout 사용하면? 기본적으로 NN 동작 자체가 변한다. Dropout을 사용하면 y = fw\(x\) à y = fw\(x, z\) 로 바뀐다. Network에 random dropout mask인 z 입력이 추가된다. Test time에 random 값을 부여하는건 좋지 않으니 그 randomness를 average out 시킨다.

![](../.gitbook/assets/image%20%2892%29.png)

Dropout mask에는 4가지의 경우의 수가 존재하는데, 그 값들을 4개의 마스크에 대해 평균화 시켜준다. Test/train time 간의 기대값이 상이하다. dropout probability를 네트워크의 출력에 곱하면 test/train의 기대값이 같아진다.

![](../.gitbook/assets/image%20%28343%29.png)

Dropout을 사용하면 네트워크 출력에 dropout probability를 곱해준다.

![](../.gitbook/assets/image%20%28272%29.png)

Train time에는 임의의 값들을 0으로 만들어주고, test time에는 마지막에 dropout probability 만 곱해주면 된다. Test time에 항을 곱하고 싶지 않으면 training time에 p를 나눠주는 방법도 있다.

![](../.gitbook/assets/image%20%28243%29.png)

정리하면 dropout은 train time에는 네트워크에 randomness를 추가해 training dat에 너무 fit 하지 않게 하고 test time에는 randomness를 평균화시켜서 generalization 효과를 준다.  Batch normalization 도 이와 비슷한 동작을 한다. 실제로 BN 사용할 때는 dropout을 사용하지 않는다. BN 자체로도 충분히 regularization 효과가 있기 때문. 다른 점은 dropout에는 우리가 조절하며 쓸 수 있는 파라미터 p가 있다는 거다. BN은 없음

![](../.gitbook/assets/image%20%2818%29.png)

Data augmentation

Train 시 사용했던 이미지를 무작위로 변환시켜서 사용한다. 원본 이미지를 하긋ㅂ하는게 아니라 무작위로 변환시킨 이미지를 학습하는 거다. Flip 하거나 crop하거나

![](../.gitbook/assets/image%20%28150%29.png)

이미지 한장에서 10개를 잘라내서의 성능을 비교한다.

![](../.gitbook/assets/image%20%2885%29.png)

Color jitter

학습시 이미지의 contrast나 brightness르 바꿔준다 PCA 방향 고려하여 color offset조절하는 방법. Data-dependent한 방법. 자주 사용하진 않음.

Random mix/combination : translation, rotation ,stretching, shearing, lens distortion

Train time에 입력데이터에 임의의 변환을 시켜주면 일종의 regularization 효과를 얻을 수 있다.

Dropout, bn, data augmention 외에도 dropconnect, fractional max pooling, stochastic depth 등의 regularizaiotn 방법들이 있따.

![](../.gitbook/assets/image%20%2878%29.png)

Transfer learing

Overfitting이 일어날 수 있는 상황 중 하나는 바로 충분한 데이터가 없을 때이다.

1.     아주 큰 데이터셋으로 모델을 한번 traing 시킨다

2.     학습된 feature를 우리가 가진 작은 데이터셋에 적용한다. FC Layer는 최종 feature와 class scores 간의 연결인데, 이 부분을 초기화 시킨다. 방금 정의한 가중치는 reinitiali하고 나머지는 freeze한다. ㅇ마지막 레이어만 가지고 모델을 학습시키는 거다.

3.     데이터가 좀 더 있따면 전체 네트워크를 fine tunignㅎ ㅏㄹ 수 있다. 네트워크 이루바가 아닌 전체 학습을 고려해 볼 수 있다. 네트워크의 더 많은 부분을 업데이트 시킬 수 있다. 보통 기존의 learning rate보다 낮춰서 학습시킨다. 기존의 가중치들이 이미 잘 학습되어 있기 때문.

![](../.gitbook/assets/image%20%28232%29.png)

-       현재 데이터 셋이 학습데이터셋과 유사하지만 소량인 경우 : 마지막 레이어만 학습시킨다

-                                                   많은 경우 : 모델 전체를 fine tuning

-                                             다르고 소량이면 : 더 많으 부분은 다시 초기화 시켜야 하거나

-                                                    많은 경우 : fine tuning

![](../.gitbook/assets/image%20%2823%29.png)

대부분 imageNEt pretrained-model을 사용하고 현재 본인의 task에 맞게 fine tuning하는 식으로 쓴다. Captionaig은 word vector를 pretrained word vector도 같이 이용할 수 있따.

 처음부터 CNN을 학습시키려고 하면 너무 시간이 오래 걸리고 효율이 떨어지니까 성능이 입증된 CNN 모델을 가져가다 feature를 추출하고, 이를 바탕으로 우리가 원하는 Classification을 수행하도록 만드는 것이다. 실질적으로 전체 CNN을 처음부터 끝까지 학습시키기에는 데이터셋이 너무 크기 때문에 불가능하다. 이미 대규모 데이터를 대상으로 학습이 끝난 ConvNet을 가져다가 초기값으로 사용하거나 fixed feature extractor로 사용할 수 있다.  
  
 - **ConvNet as fixed feature extractor**   
: ConvNet의 끝에 있는 classification layer를 제거하고 convolutional layer를 통해 처리 되는 값을 얻으면 완전한 feature extractor가 된다.\(이렇게 얻어진 feature를 CNN codes라고 함\) 이 CNN codes와새로운 training set을 사용해 linear classifier\(ex, Linear SVM or Softmax\)를 학습한다.  
     -&gt; 마지막의 Classification layer만 retrain  
  
 - **Fine-tuning the ConvNet**   
: 끝의 fully-connected layer도 없애고 앞 단의 convolutional layer를 새로운 데이터로 다시 학습시켜서 역전파를 통해 weight를 업데이트 한다. training 데이터가 많을 때 사용할 수 있는 방법이다. 경우에 따라 앞쪽 레이어는 고정시키고 뒤쪽 레이어만 fine-tuning 하기도 한다. \(앞단의 레이어를 통해 얻어지는 것은 직선이나 곡선같은 general한 feature를 학습하지만, 뒷단의 고차원적 feature는 특정 도메인에 종속 될 수 있다\)  
     -&gt; pretrain된 전체 네트워크를 재조정\(fine-tuning\)  
  
Q\) 언제, 어떻게 transfer learning 을 할 것인가?  
 - 새 데이터가 작지만 원래 데이터와 비슷한 경우 : CNN codes를 이용해서 linear classifier를 학습  
 - 새 데이터가 크고 원래 데이터와 비슷한 경우 : 데이터가 많으니까 fine-tuning through the full network  
 - 새 데이터가 작고 원래 데이터와 많이 다른 경우 : 데이터가 적으니까 분류기만 학습. 데이터셋 자체가 많이 다르니까 ConvNet을 모두 통과한 결과보다는 앞 쪽 레이어의 값들을 사용해서 분류기를 학습시킴.  
 - 새 데이터가 크고 원래 데이터와 많이 다른 경우 : 데이터가 많으니까 처음부터 CNN을 구축해도 됨. 그래도 pretrained model의 weight로 초기값을 설정하고 학습시키는게 더 낫다. convolution layer를 처음부터 끝까지 fine-tuning 하는 것도 가능함.  


