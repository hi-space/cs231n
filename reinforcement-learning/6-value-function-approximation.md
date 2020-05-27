# 6. Value Function Approximation

여태 배운 것들은 방법들은 table lookup 을 이용해 각 칸들의 값들을 update 시켜봤다. 하지만 현실의 문제는 table 형식으로 표현되지 않을 정도로 state가 많아서 모든 값들을 저장하고 update 시키기는 어렵다. 그래서 이번엔 table lookup을 사용하지 않고 prediction하고 control 할 수 있는 방법에 대해 알아보려고 한다.

## Introduction

Value function은 state의 갯수만큼, Q 에서는 state-action fair의 갯수만큼 table 공간이 필요했다. 큰 MDP 문제에서는 메모리에 저장하기에 너무 많은 state와 action들이 존재했다. 그래서 value function을 function approximation을 해서 estimate 하는 방식을 사용하려고 한다.

$$
\hat{v}(s, w) \approx v_{\pi}(s) \\
\hat{q}(s, a, w) \approx q_{\pi}(s, a)
$$

* $$ v_{\pi}(s)$$ : 실제 value function
* $$\hat{v}(s, w)$$ : 실제 value 값을 모방하는 approximation function
* $$w$$: $$\hat{v}$$함수 안에 포함되어 있는 parameter 값
* $$\approx$$ : approximation

function approximation을 하게 되면 봤던 state 뿐만 아니라 보지 못한 state에 대해서도 generalize가 잘된다. MC나 TD learning을 통해 파라미터 $$w$$를 업데이트하며 학습해가는 방식이다.

![Types of Value Function Approximation](../.gitbook/assets/image%20%28433%29.png)

일반적인 함수의 모양을 나타내는 black box 이다. 어떤 input 값을 받았을 때 internal parameter인 $$w$$와 함께 연산되어 그에 해당하는 output을 산출한다.

1 그림\)  value function으로, $$s$$를 input으로 넣으면 $$w$$값들을 통해 $$\hat{v}(s, w)$$의 output이 나온다.

2, 3 그림\) Q 는 두가지 형태로 함수를 만들 수 있다.

* \(action in 형태\) $$s$$와 $$a$$를 input으로 넣으면 $$\hat{q}(s, a, w)$$이 output으로 나온다.
* \(action out 형태\) $$s$$를 input으로 넣으면 $$s$$에서 할 수 있는 모든 action에 대해서 output이 나온다.

![Which Function Approximator?](../.gitbook/assets/image%20%28434%29.png)

function approximation으로 사용할 수 있는 함수는 liner combinations, neural network, decision tree 등 여러가지가 될 수 있겠지만, 그 중에서도 미분가능한\(differentiable\) 함수를 사용할거다. 그래야만 그 상태의 gradient를 구해서 update 할 수 있기 때문이다. 

non-stationary, non-iid : 모분포가 계속 바뀌고 independent 하지 않아서 이전의 값들이 이후의 값에 영향을 미침 \(??

## Incremental Methods

### Gradient Descent

J라는 함수가 있다 .w 값이 input으로 들어가면 output이 나오는 함수. w는 n차원 벡터 J라는 함수를 최소화하는 input w를 찾고 싶다. w 찾을 때 사용하는 것이 gradient descent 방법.

convex하다. local minimum= global minimum

w가 벡터니까 차례대로 w\_1, w\_2이 있을 때 J의 gradient 를 구하게 되면, J가 가장 빠르게 변하는 방향이 나온다. 그 방향으로 a 값만큼 조금 움직인다. 가파른 방향으로 조금씩 움직여서 가장 작은 값으로 간다.

### Value function approx

목적함수 J를 줄이고 싶다. value function을 잘 학습하는게 목적이다. v^이 잘 모방하는게 목표. 그 차이가 작을 수록 좋다. oracle이 있어서 true value function 을 안다고 가정. 실제 value function 과 모방한 value function 의 오차를 최대한 작게 하는 것을 J의 목적 함수로 정한다. \(J를 loss라고 보면 됨\) J를 줄이는 방향으로 w를 업데이트 해야한다.

w를 저만큼 update 하면 된다.

stochastic gradient는 gradient의 sample 들을 뽑아서 넣어주는 거. \(방문했던 state들을 input으로 넣어주는 것\) 여러번 반복하게 되면 expected update와 full gradient update와 동일하게 된다.

### 

일반적인 이야기였음. 이번엔 Linear 에대해 이야기 해보자.

state가 S가 있으면 n개의 feature를 만들 수 있다.

이상한 값들이 나올텐데 feature vector들에 w를 내적곱해서 value function 이 나온다.

objective function는 true value function과 모방함수에 Transpose

각 feature 마다 가중치를 줘야하니까 w도 n개가 있는거.

stochatic에서는 global optimum에 수렴한다. linear 함수이기 때문에 최저점이 하나 밖에 없기 때문.

--

table lookup은 linear value function의 하나의 예시라고 볼 수 있다.

--

실제 문제에서는 supervisor가 없기 때문에 true value function은 모른다. 그 true value function 자리에 MC나 TD를 위치시키면 된다.

prediction 문제. cumulative reward에 대해 예측을 어떻게 하는지. value function approximation과 실제 value 사이에 제곱을 minimize 하기 위한 것.

### MC

value는 return의 기댓값. sampling할 때마다 다른 episode가 나오고 다른 return 값이 나온다. 그 데이터를 통해서 update 해도 된다. local optimum에서도 수렴한다. non-linear에서도.

### TD

TD error : delta linear TD\(0\)는 global optimum에 가깝게 수렴한다.

MC는 unbiased 하다. variance가 커도 맞춘다. TD는 꼭 그 bias가 아니어도 vairnace가 작기 대문에 global optimum에 수렴해 갈 수 있다.

## Incremental Control Algorithm.

GPI.. 이용.. policy 찾는거.. imporve는 epsilon으로 평가는 approximation.. q 에대해서.. \(model free니까\). Q를 끼워넣으면 policy를 학습한다.

V나 Q 나 똑같다. V를 Q로만 바꿔주면 된다.

Linear SARSsa linear한 function approximation을 썼음.

MC에서만 수렴한다고 되어 있는데 실제로는 TD\(0\)나 람다나 잘 수렴한다.

## Batch Methods

Incremental 은 gradient descent를 이용해 sample 하나를 뽑아서 그거롤 update 하고 극걸로 policy update 하고. 한번 update하고 그 경험은 버려지니까 그 sample이 효과적으로 사용되지 않는다. 쌓여진 경험 데이터르를 re-use 하면서 학습하는거다.

incremental은 경험을 쓰고 버리고, batch는 경험을 쌓아놓고 쓰고. off-policy느낌이라고 보면 됨

incremental은 \pi를 따라가면서 했는데 이건 주어진 D를 따라가면서 하는거. D에서 s, value를 smapling 해서 그걸로 gradient descent 한다. 데이터를 좀 더 효율적으로 쓸 수 있다. 이를 experience replay 라고 한다. off-policy할 때 많이 사용되는 방법이다.

non-lenear할 때 수렴성을 높이기 위해 experien replay : transitoin들을 replay memory에 쌓고 랜덤하게 미니배치를 뽑아서, 그걸로 학습을 한다. fixed Q-targets : TD target을 계산할 때 파라미터를 고정시켜놓고 하다가, 업데이트. target network를 두개를 이용하는 거.

## Batch Methods

