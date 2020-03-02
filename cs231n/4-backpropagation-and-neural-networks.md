# 4. Backpropagation and Neural Networks

이전 강의에서 진행했던 Loss Function과 Optimization에 대해 간략히 정리해보자.

![Loss Function](../.gitbook/assets/image%20%28254%29.png)

* Input 값은 x와 W, Output은 분류하고자 하는 클래스들에 대한 score vector.
* 최적의 loss를 갖게 하는 파라미터 W를 찾기 위해 loss function의 W에 관한 gradient를 계산해서 찾았다.
* Regularization term은 우리의 모델이 얼마나 단순한 지, 일반화 하기에 적합한 단순한 모델인지를 표현해준다.

## Backpropagation

![Gradient Descent](../.gitbook/assets/image%20%287%29.png)

 finite difference approximation을 이용해 gradient를 구하는 방법이 있었는데, 이는 매우 느리고 근사치이지만 직접 써내려가며 하기엔 가장 쉬운 방법이다. 하지만 파라미터가 많아지고 관계식이 복잡해질 수록 최종 loss에 대한 각 파라미터들의 미분값을 구하는 것이 힘들어진다.

analytic gradient 사용하는 것은 빠르고 정확하지만, 이 식을 유도하기 위해 많은 수학과 함께 미적분학을 해야 한다. 또한 실수하기가 쉽다. 앞으로 대부분 analytic gradient 방법으로 진행하게 될 텐데, 이를 사용한 응용에 대해서 수학적 검증이 필요하다.

어떤 복잡한 함수의 analytic gradient를 어떻게 계산할 것인가?

![Linear Classifier - Computational graph](../.gitbook/assets/image%20%28274%29.png)

위의 그림은 Linear Classifier를 Computational graph로 표현한 것이다.

* 곱셈 노드는 행렬 곱셈, W와 x의 곱셈은 score vector를 출력
* loss의 L은 hinge loss 계산 노드를 거친 data loss와 regularization항의 합

computational graph를 모양으로 나타냄으로써 backpropagation 할 수 있게 구성한다. backpropagation은 gradient를 얻기 위해 computational graph 내부의 모두 변수에 대해 chain rule을 재귀적으로 사용한다.

### Chain Rule

gradient descent에서 관계식이 복잡하면 각 파라미터들의 미분값을 구하는게 힘들어진다. 

$$
y=x*g ~~~~~~~~~~~~~
 g=x^2
$$

예를 들어 위와 같이 어떤 변수와 변수간의 상관관계들로 서로가 연결되어 있고 알고 있는 것은 y로부터의 loss function 밖에 없을 때와 같은 상황을 관계식이 복잡하다고 얘기한다. 

위의 식도 단순히 한다리만 건너면 input x 값에 도달할 수 있지만 이 관계식이 몇십개가 된다면 모든 파라미터들에 대해서 y에 대한 loss 값을 구하기가 힘들어질 것이다. 이를 좀 더 쉽게 계산하기 위해서 Chain rule을 이용한다. 

![Chain Rule](../.gitbook/assets/image%20%28205%29.png)

df/dx 를 구하기 위해서는 df/dg \* dg/dx를 하면 된다. 

f가 loss function이라고 하고 x가 loss에 대한 gradient를 계산하고 싶은 파라미터라면 \(df/dx를 계산\) 계산되어 있는 df/dg 와 dg/dx를 알고 있기만 하면 이를 쭉 이어 붙여서 곱하면 된다.

### Backpropagation

맨 마지막 Loss에 대한 미분값만 구하면 Chain Rule을 이용해서 그 이전 노드\(파라미터\)들에 대해서 Loss에 대한 미분값을 구할 수 있게 된다. 이렇게 미분값을 뒤로 곱해서 나아가는 것을 Backpropagation 이라고 한다.

![simple example](../.gitbook/assets/image%20%2825%29.png)

Backpropagation에 대한 simple 한 example이다. chain rule을 이용해 뒤쪽 노드부터 앞쪽 노드까지의 gradient를 계산한다. 만약 y의 값을 변화시키면 f 값은 그것의 영향력 만큼 변할 것이다.

![](../.gitbook/assets/image%20%28299%29.png)

각 노드들은 오직 주변에 대해서만 알고 있다. 우리가 가지고 있는건 각 노드\(f\)와 각 노드의 local 입력\(x, y\)이다. 입력\(x, y\)은 이 노드\(f\)와 연결되어 있고, 이 값은 이 노드를 통해 출력값\(z\)을 얻게 된다. 그리고 우리는 이 노드에서 local gradient를 구할 수 있다.

![](../.gitbook/assets/image%20%28223%29.png)

각 노드는 local 입력을 받고 다음 노드로 출력값을 보낸다. 그리고 우리가 계산한 local gradient는 들어오는 입력에 대한 출력의  기울기\(dL/dz\)이다. 

Backpropagation은 그래프의 뒤에서부터 시작부분까지 진행되는데, 각 노드에 도달하면 출력과 관련한 노드의 gradient가 인접해 있는 노드로 전파된다. 이 노드에 도달할 때 까지 z에 대한 최종 loss 은 이미 계산되어 있고, 그 다음은 x, y의 값에 대한 바로 직전 노드의 gradient 를 찾고자 한다. 이는 chain rule을 통해 가능하다. x의 gradient는 z에 대한 gradient와 x에 대한 z의 local gradient로 합성된다.

정리하면, 출력\(z\)의 gradient를 local gradient와 곱해서 노드의 입력\(x, y\)에 대한 gradient를 구한다. 이렇게 각 노드는 계산된 local gradient를 가지고 있다. 값들은 또다시 상위 노드로 전달될 것이고, 이 값들을 각 local gradient와 곱하기만 하면 된다. 연결되어 있는 노드에만 신경써서 계산하면 된다.

### Modularized implementation

각 노드들에 대해서 local 하게 보며 upstream gradient와 chain rule을 이용해 local gradient를 계산해봤다. 이것을 forward pass, backward pass API로 생각해 볼 수 있다. Forward pass 에서는 노드의 출력을 계산하는 함수를 구현하고, Backward pass에서는 gradient를 계산한다.

```python
class ComputationalGraph(object):
    def forward(inputs):
        # 1. [pass inputs to input gates...]
        # 2. forward the computational graph:
        for gate in self.graph.nodes_topologically_sorted():
            gate.forward()
        # the final gate in the graph outputs the loss
        return loss 
    def backward():
        for gate in reversed(self.graph.nodes_topologically_sorted()):
            # little piece of backup (chain rule applied)
            gate.backward()
        return inputs_gradients        
```

Forward는 노드를 처리하기 이전에 노드에 들어오는 모든 입력값들을 처리하고, Backward는 역순서로 모든 게이트를 통과한 다음에 게이트 각각을 거꾸로 호출한다.

![\(x \* y = z\) Example](../.gitbook/assets/image%20%2888%29.png)

```python
class MultiplyGate(object):
    def forward(x, y):
        z = x*y
        self.x = x # must keep these around
        self.y = y
        return z
    def backward(dz):
        dx = self.y * dz # [dz/dx * dL/dz]
        dy = self.x * dz # [dz/dy * dL/dz]
        # self.x : Local gradient
        # dz = Upstream gradient variable
        return [dx, dy]
```

여기서 중요한 것은, forward pass의 값을 cache에 저장하는 것이다. forward pass가 끝나고 나서 backward pass에서 chain rule을 계산할 때 많이 사용되기 때문이다.

### Summary

우리가 사용할 네트워크는 매우 복잡하기 때문에 모든 파라미터에 대해 gradient를 손으로 일일이 구하는 것은 비현실적이다. 그래서 gradient를 계산하기 위해  Computational graph에서 chain rule을 재귀적으로 이용하는 backpropagation을 사용한다. 이 과정에서 각 노드의 입력이나 파라미터 등 모든 중간 변수들이 구해진다.

* Forward pass : 연산결과를 계산하고 결과를 저장 \(gradient를 계산할 때 backward pass에서 chain rule로 계산할 때 사용\)
* Backward pass : upstream gradient와 저장한 값들을 곱해 각 노드의 input에 대한 gradient를 구하고, 그 값을 연결되어 있는 이전 노드로 통과시킨다.

## Neural networks



![Neuron, Computational graph](../.gitbook/assets/image%20%28120%29.png)

Computational node는 Neuron이 동작하는 것과 비슷한 방식으로 볼 수 있다.

![Exampel feed-forward computation of a neural network](../.gitbook/assets/image%20%28189%29.png)

```python
# forward-pass of a 3-layer neural network
# activation function (sigmoid)
f = lambda x: 1.0/(1.0 + np.exp(-x)) 
# random input vector of three numbers (3x1)
x = np.random.randn(3, 1)
# calculate first hidden layer activations (4x1)
h1 = f(np.dot(W1, x) + b1)
# calculate second hidden layer activations (4x1)
h2 = f(np.dot(W2, h1) + b2)
# output neuron (1x1)
out = np.dot(W3, h2) + b3
```

