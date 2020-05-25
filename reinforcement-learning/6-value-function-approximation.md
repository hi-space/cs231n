# 6. Value Function Approximation

여태 배운 것은 작은 문제에 대한 방법들이였다. 이제 현실에 적용하기 위한 큰 문제들을 풀어볼거다. table lookup을 이용해 값들을 update 시켜봤었는데, 이번엔 approximate 이용해볼거다.

table lookup은 각 칸에 대해 초기화 하고 그 칸의 값들을 update 했다. 현실의 문제에서는 state가 굉장히 많아서 모든 state에 대한 값을을 저장하고 update 하기 어렵다. 이런 문제들에서 어떻게 prediction 하고 control 할 수 있을까?

Value Function Approximation

## Introduction

value function은 sate s 갯수만큼, Q 에서는 state-action fair 갯수만큼 빈칸이 필요했다.

function approximation이라는 개념이 나온다. v\_pi\(s\)는 실제 value 라고 하면 v 햇 은 approximation 이다. 모방하는 함수. w 는 v햇 함수 안에 들어있는 parameter 값.

봤던 state 부터 보지 못한 state에 대해 generalize가 잘 된다. -&gt; 보지 못한 state에 대해서도 알맞은 output을 만들어준다. 학습한다는 것은 w를 update 한다는 거.

--

black box 인 function 이 있다.

value function\) s를 input을 넣으면 w 라는 값들통해 v 햇 \(s, w\) 의 output 이 나온다.

action-value function\) Q 두가지 형태로 함수를 만들어줄 수 있다. 1. s와 a를 input으로 넣으면 q^\(s, a, w\) output이 나온다. \(action in 형태\) 2. s를 input으로 넣으면 s에서 할 수 있는 모든 action에 대해서 output이 나올 수도 있다. \(action out 형태\)

w는 internal parameter.

--

fuction approi 함수는 뭘 쓸 수 있을까?

* linear, 가중 합을 이용
* neural network
* decision tree 등등

이중에서 differentiable한 function approximator 를 사용할 거다. 미분 가능한. 그래야 gradient를 구해서 update 할 수 있기 때문. non-stationary, non-iid 모분포가 계속 바뀌고 independent 하지도 않다\(이전의 값들이 이후 값에 영향을 미침\)

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

