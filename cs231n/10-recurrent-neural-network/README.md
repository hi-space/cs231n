# 10. Recurrent Neural Network

![](../../.gitbook/assets/image%20%283%29.png)

RNN은 네트워크의 입/출력 가변적일 수 있다.

고정 길이의 입/출력일 때도 sequential processing이 되는 경우에도 유용하게 사용 될 수 있다. 이미지의 정답을 feed forward 한번으로 알아 내는게 아니라, 점차적으로 여러 개 확인한 후에 최종 판단을 한다.

동일한 방법으로 이미지 이미지들을 바탕으로 새로운 이미지들을 만들어 낼 수 있다. 순차적으로 전체 출력의 일부분을 차례로 생성해 나간다.

![](../../.gitbook/assets/image%20%28143%29.png)

일반적으로 RNN은 작은 Recurrent core cell 을 가지고 있다. 그리고 내부에 hidden state를 가지고 있다. Hidden state는 RNN이 새로운 입력을 불러들일 때 마다 새로 업데이트 된다. Hidden state는 모델에 feed back 되고 이후에 또 다시 새로운 입력 x가 된다.

1.     입력을 받는다

2.     Hidden state를 업데이트한다

3.     출력값을 내보낸다

![](../../.gitbook/assets/image%20%28268%29.png)

RNN block은 재귀적인 관계를 표현할 수 있게 된다.

함수 f는

H\_t-1 : 이전 상태의 hidden state

X\_t : 현재 상태의 입력

를 받아 h\_t를 출력 \(다음상태의 hidden state\)

RNN에서 출력 y를 가지려면 h\_t를 입력으로 하는 FC layer를 추가해야 한다. FC layer는 매번 업데이트되는 hidden state를 기반으로 업데이트 된다.

여기서 중요한건 함수 f와 파라미터W는 매 스텝 동일하다는 거다.

![](../../.gitbook/assets/image%20%2865%29.png)

수식으로 표현하면 다음과 같음

왜 tanh를 쓰는건가? 다른 non-linear가 아닌.. lstm 때 다시 배우자

-       RNN이 hidden state를 가지며 이를 재귀적으로 feed back 한다

Multiple time steps을 unrolling 하면 좀 더 명확히 볼 수 있다.

![](../../.gitbook/assets/image%20%2899%29.png)

-       동일한 가중치 행렬 W가 매번 사용된다. H와 x는 달라지지만 W는 동일하다.

각 스텝에서의 W에 대한 그래디언트를 전부 계산한 뒤에 이 값들을 모두 더해주면 backporp을 위한 행렬W의 gradient가 된다.

각 스텝마다 y가 나오고 그에 대한 loss를 계산할 수 있다. RNN의 최종 loss는 이런 loss들의 합이다.

모델을 학습시키려면 dLoss/dW를 구해야 한다. Loss flowing은 각 스텝마다 일어난다. 가 스텝마다 W의 gradient를 구할 수 있다.

![](../../.gitbook/assets/image%20%28187%29.png)

Many to one 모델인 경우에는 최종 hidden state에서만 결과 값이 나올 거다. 최종 hidden state가 전체 시퀀스의 내용에 대항 일종의 요약으로 볼 수 있다.

![](../../.gitbook/assets/image%20%28254%29.png)

One to many 모델

이 경우에 고정 입력\(x\)는 모델의 initial hidden state를 초기화 시키는 용도로 사용된다. 그리고 모든 스텝에서 출력값을 가진다.

![](../../.gitbook/assets/image%20%28188%29.png)

Seq to seq 모델

Many to one과 one to many 모델의 ruifgkq

Encoder, decoder 두개의 스테이지로 연결되는 거다. Encoder는 가변 입력을 받아 final hidden state를 토앻 전체 sentence를 요약한다. encoder에서는 many to one을 수행한다.

decoder에서는 one to mayn 수행. 하나의 벡트 가 입력. 그리고 가변 출력은 매 스텝 적절한 답을 출력할 거다. Ex, machine translation

output sentence의 각 loss들을 합해서 backprop을 진행한다.

![](../../.gitbook/assets/image%20%2811%29.png)

Softmax의 확률 분포에서 가장 높은 스코어를 선택하지 않고 확률분포에서 샘플링 한다. \(그냥 높은 스코어를 선택할 수도 있다\) argmax 정책을 사용하면 안정적일 순 있지만, 확률 분포에서 샘플링하는 방법을 선택하면 모델에서의 다양성을 얻을 수 있다.

Gradient 업데이트 과정이 매우 느리고 모델이 수렵되지 않는다. 메모리 사용량도 클 것 이다.

실제로는 truncated backprop을 통해 backprop을 근사시키는 기법을 사용한다. Train time에 한 스텝을 일정 단위로 잘라서 forward pass하고 sub sequence의 loss를 계산하고 gadietn step 을 진행한다. 이 과정을 반복해 나가는데, 다만 이전 batch에서 계산한 hidden states는 계쏙 유지한다. 다음 batch의 forward pass를 계산할 때는 이전 hidden state를 이용하지만 gradient step은 현재 batch에서만 진행한다.

Stochastic gradient descent의 시퀀스 데이터 버전이라고 볼 수 있다.

Image Captioning Example

![](../../.gitbook/assets/image%20%28285%29.png)

기존의 classification 네트워크에서 class를 구분하는 레이어인 마지막 FC와 softmax 레이어를 제거하고 4096벡터를 입력으로 사용한다.

![](../../.gitbook/assets/image%20%28299%29.png)

이전까지는 RNN 모델이 두개의 가중치 행렬을 입력으로 받았따 \(현재 스텝의 입력, 이전스텝의 hidden state\) 이제는 이미지 정보도 추가한다. 세번째 가중치 행렬을 추가하는 거다. Hidden state를 계산할 때 마다 모든 스텝에 이 이미지 정보를 추가한다.

![](../../.gitbook/assets/image%20%28234%29.png)

샘플링 된 단어 y0가 들어가면 그것이 바로 다음 input으로 들어가고, 그 다음 단어를 만들어 간다. 모든 스텝이 종료되면 한 문장이 만들어지게 된다. End token을 만나면 문장이 종료된다.

Train time에는 ㅁ hems caption의 종료 지점에 token을 삽입한다. 그래야 네트워크가 학습하는 동안에 시퀀스의 끝에 token 이 나와야 한다는 것을 학습하기 때문이다. 학습이 끝나고 test time에는 모델이 문장 생성을 끝마치면 토큰을 샘플링 한다.

하지만 supervised learning의 한계인 것 처럼, train time에 학습하지 못한 부분에 대해서는 취약하다.

![](../../.gitbook/assets/image%20%28112%29.png)

조금 더 진보된 attention 모델. Captionaing 할 대 이미지의 다양한 부분을 attention해서 볼 수 있다.

CNN으로 벡터 하나를 만드는게 아니라 각 벡터가 공간 정보를 가지고 있는 grid of vector를 만들어낸다. \(LxD\) forward pass시에 매 스텝 vocabulary에서 샘플링 할 때 모델이 이미지에서 보고 싶은 위치에 대한 분포 또한 만들어낸다. 이미지의 각 위치에 대한 분포는 train time에 모델이 어느 위치를 봐야하는 지에 대한 attention이다.

![](../../.gitbook/assets/image%20%28129%29.png)

![](../../.gitbook/assets/image%20%2836%29.png)

![](../../.gitbook/assets/image%20%28111%29.png)

첫번째 hiddn eats는 이미지 위치에 대한 distribution을 계산한다.\(a1\) 이 distribution을 다시 벡터 집합\(LxD feature\)와 연상해서 이미지 attention\(z1\)을 생성한다. 이 요약된 벡터 z1은 다음 스텝의 입력으로 들어간다. 그리고 두개의 출력이 생성된다. \(d1: vocabulary의 각 단어들의 분포, a2 : 이미지 위치에 대한 분포\)

이 ㄱ과정을 반보갛면 매 스텝마다 a,d 가 만들어진다.

Train을 마치면 모델이 caption 생성하기 위해 이미지의 attention을 이동시키는 걸 확인할 수 있다.

-       Soft attention : 모든 특징과 모든 이미지 위치 간의 weighted combination을 취하는 경우

-       Hard attention : 모델이 각 타임 스텝마다 단 한 곳만 보도록 강제한 경우

50분

