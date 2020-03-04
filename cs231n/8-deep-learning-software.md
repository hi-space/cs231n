# 8. Deep Learning Software

![](../.gitbook/assets/image%20%28198%29.png)

FC Layer + relu

![](../.gitbook/assets/image%20%28218%29.png)

Placeholder : Input variable.

메모리 할당이 일어나지 않고 단순 그래프만 구성

Session 에 들어가기 위해 실질적인 데이터를 만들어야 한다. Numpy를 이용해 값을 할당함.

Palcehodler : 그래프 밖에서 데이터를 넣어주는 변수

Variable : 그래프 내부에 있는 변수

![](../.gitbook/assets/image%20%28189%29.png)

Pytorch는 세가지 추상화 레벨을 정의 해놨다. Tensor\(명령형 imperative 배열, gpu에서 수행\) variable\(그래프의 노드. 그래프를 구성하고 그레디언트 등을 계싼\) , module

pytorch에는 tensor obecjt가 있따. Tensorflow의 numpy array와 유사.

고수준의 추상화를 이미 내장하고 있어, tensorflow처럼 어떤 모듈을 선택할 지 고민할 필요 없다. 그냥 module 객체만 사용하면 된다.

![](../.gitbook/assets/image%20%28136%29.png)

Pytorch tensor와 numpy 가장 큰 차이점은 pytorch tensor는 GPU에서도 돌아간다는 점이다. 데이터 타입만 바꿔주면 된다. Torch를 Numpy + GPu라고 보면 되다.

![](../.gitbook/assets/image%20%28221%29.png)

Variable은 computatinon graph만들고 이를 통해 gradient 자동 계싼한다. Variable 자체에 gradient가 포함되어 있다.

ㅅtensorflow와 pytorch의 차이는?

Tensorflow는 그래프를 명시적으로 구성한 다음에 그래프를 돌렸따. Pytorch는 forward pass할 때 마다 매번 그래프를 다시 구성한다.

![](../.gitbook/assets/image%20%28127%29.png)

직접 autograd 함수를 구현할 수도 있다. 하지만 대부분은 이미 구현되어 있다.

![](../.gitbook/assets/image%20%2899%29.png)

Tensorflow의 경우는 keras 같은 라이브러리가 higher level ap를 제공해줬따. Pytorch는 nn package가 담당한다.

![](../.gitbook/assets/image%20%28174%29.png)

Optimizer 객체를 구성해 놔서, 모델에게 파라미터를 optimize 하고 싶다고 쓸 수 있다.

![](../.gitbook/assets/image%20%28205%29.png)

전체 네트워크 모델이 정의되어 있는 class를 nn module class로 작성해야 한다.  Module은 일종의 네트워크 레이어.

![](../.gitbook/assets/image%20%28305%29.png)

Dataloader는 minibatches를 관리한다. Multi-threaidn 을 통해 데이터를 가져오는 것을 알아서 관리해준다. 실제 데이터를 이용하고자 할 때 데이터를 어떤 방식으로 읽을 것인지를 명시하는 dataset class만 작성해주면 dtalaoader로 wrapping 시켜서 학습 할 수 있다.

![](../.gitbook/assets/image%20%28154%29.png)

내붖적으로 data shuffling, multithreaded dataloading과 같은 것들을 알아서 관리해준다.

![](../.gitbook/assets/image%20%2873%29.png)

Pretrained model을 쓸 수도 있음

Visdom은 학습되는 데이터들을 시각화해준다. tensorboard는 computational graph시각화를 제공하지만 visdom은 제공 노노.

![](../.gitbook/assets/image%20%28214%29.png)

TF는 그래프를 구성하고, 그래프를 반복적으로 돌리는 두단계로 나뉜다. 그래프가 단 하나만 고정적으로 존재하기 때문에 static computational graph라고 한다.

Pytorch는 forward pass할 때마다 그래프를 새로 그리기 때문에 dynamic computational graph라고 한다.

![](../.gitbook/assets/image%20%28209%29.png)

Static graph관점에서 그래프를 한번 구성해놓으면 학습시에 똑 같은 그래프르 아주 많이 재사용 하게 된다. 일부 연산들을 합치고 재배열 시키는 등 효율적으로 연산을 할 수 있도록 최적화시킬 수 있다.  코드가 더 효율적으로 동작한다.

![](../.gitbook/assets/image%20%2816%29.png)

그래프를 한번 구성해놓으면 메모리 내에 그 네트워크 구조를 가지고 있다. 그 자체를 disk에 저장하 ㄹ 수 있다. 전체 네트워크 구조를 파일로 저장하고, 그래프를 다시 불러올 수도 있다.

Dynamic은 그래프 구성, 실행 과정이 엮겨있기 때문에 모델을 제사용 하기 위해 항상 원본 파일이 필요하다.

![](../.gitbook/assets/image%20%28263%29.png)

Dyanamic은 매번 새로운 그래프를 그려주기 때문에 분기가 생겼을 때 forward pass에 적절한 하나를 선택해서 새로운 그래프를 만들어 주면 된다. Tensorflow 같은 경우는 그래프를 하나 더 만들어야 한다. 조건부 연산을 명시적으로 정의하는 control flow operator를 추가해야 한다.\(cond\) 가능한 control flow를 미리 고려해 그래프 내에 한번에 넣어줘야 한다.  단순 python 문법으로는 불가능하고 특정 tensorflow 함수가 필요하다.

![](../.gitbook/assets/image%20%28131%29.png)

Loops

재귀적은 연사을 할 때 데이터 sequence가 다양한 사이즈일 수 있다.

![](../.gitbook/assets/image%20%28216%29.png)

Pytorch는 기본 for loop 사용하면 된다. 데이터 사이즈에 맞는 적절한 그래프를 손쉽게 만들 수 있다. 하지만 tensorflow에서는 그래프를 미리 만들어줘야하기 때문에 명시적으로 그래프에 loop을 넣어줘야 한다. \(foldl\)

Computational graphg의 전체 흐름을 다 구현해놔아ㅑ 한다. Control flow나 모든 데이터 구조 등을 다 구현해야 한다. 하지만 pytorch는 pyhon 명령어를 통해 가능하다.

Tensorflwo에도 dynamic graph를 작성하는 TF 라이브러리가 있다ㅏ.  사실은 static graph로 만든 트릭이다. Dynamic graph역할을 더해줄 수 있찌만 pytorch dynamic 보다는 부자연스럽다.

Dynamicgrahps는 recurrent network, recursive network, modular network 등의 application 에 적합하다.

![](../.gitbook/assets/image%20%28280%29.png)

정리하면 이렇다. 철학이 다르다.

Google은 딥러닝 필요한 모든 곳에서 동작하는 프레임워크를 만드는 것이 목적잉다. TF 하나로 distributed systems, production, deployment, mobile, research를 다 커버하도록.

Facebook은 research에 특화 되어 있다. 연구 사이클을 단축하도록. Pytorfch는 production 개발을 위한 지원이 많진 않다. 대신 caffe2가 그 역할을 담당한다.

Tensorflow는 어떤 환경에서든 잘 동작하는 만능 프레임워크다. 하지만 higherlevel wrapper를 섞어 쓰거나 dyanamcic 을 사용해얗하는 경우는 불편할 수 있다.

연구 목적으로 사용하면 pytorch가 최고 좋다. 하지만 코드가 적다. Tensorf가 많다

제품 배포 자체가 목적인면 caffe2, tensorflow가 좋다. 특히 모바일 디바이스라면.

