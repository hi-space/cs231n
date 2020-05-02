# Markov Decision Processes

## Markov Processes

환경이 fully observable 할 때 MDP 가 된다. 모든 강화학습 문제는 MDP 형태로 만들 수 있다.

### Markov Property

![Markov](../.gitbook/assets/image%20%28261%29.png)

{% hint style="info" %}
The future is independent of the past given the present
{% endhint %}

* history는 필요없고 state만 필요하다. state를 알아내는 경우, history는 더이상 필요하지 않다.
* state는 미래의 충분한 통계이다.

### State Transition Matrix

state 들이 n 개 있고, discret 하게 존재한다. 매 tick마다 state를 옮겨 다니는 것 - markov process

각 state들로 전이할 확률

s에서 s'로 갈 확률

row 의 합은 1이 된다.

갈 수 있는 state들이 n, 

s 에 있을 때 다음 state\(s'\)로 이동할 확률

n x n matrix가 있겠지



memoryless random process

* memoryless : 이전의 state를 안본다. markov property
* random process : 샘플링할 수 있다. 



episode : 한번 어느 state에서 시작해서 final state 까지 가는 것. episode를 sampling 한다고 한다.



environment의 dynamics에 대해 설명한 것.

## Markov Reward Process

state, state transition matrix, reward, discount factor 로 Markov Reward Process를 표현할 수 있다.

markov process에서 reward만 추가된 거다.

state마다 reward가 있다. 확률적으로 옮겨지는 것이기 때문에 state에만 있으면 된다.



강화학습은 return을 maximize 하는 것이다. reward와는 다름.

return은 state 이동하면서 얻어진 reward의 총합인데, 단순 총합은 아니고 감마를 곱해줘서 멀리 있는 reward 값은 낮게 해서 더한다. 



discount를 하는 이유는? 

* 수학적으로 편해서. discount 를 곱해야 수렴성이 커진다. 
* 사람/동물 행동은 보통 즉각적인 reward를 선호한다.
* terminal로 가는 것이 보장된다면 감마가 1이여도 된다. \(??\)

### Value Function

return의 기댓값.

어떤 state에 왔을 때 episode 마다 return이 생길거고

G\_t는 리턴값. G\_t는 확률변수. \(샘플링에 따라 에피소드가 달라지고 그 return 값이 다를텐데, 그 return 값들을 평균낸다.\)

### Bellman Equation for MRPs

value function은 bellman equation에 의해 iterate 하게 계산된다.

## Markov Decision Process

MDP에는 Markov Reward Process에서 action이 추가된 것이다.



MRP에서는 state 에 reward가 있었는데 MDP 에서는 action 마다 reward가 주어진다. 특정 action이 항상 같은 state에 가지않고 확률적으로 어떤 state로 갈 지 정해진다. \(state의 전이 확률에 따라서\)

### Policy

MRP에서는 state간 이동을 할 때 확률분포에 의해서만 이동했다. 하지만 MDP 에서는 action에 대한 확률에 따라 state로 이동하게 된다. 특정 action을 수행하는 policy가 필요하다. agent의 행동을 결정해준다.

MDP는 environment 거고 policy는 agent거.



policy가 고정되어 있으면  ㅜㅜ?

### Value Function

MRP에서는 action이 없었기 때문에 policy가 필요없지만, MDP에서는 필요함. value function에는 state-value function과 action-value function 이 있다.

#### state-value function

에피소드를 샘플링해서 나오는 return 값의 기댓값이라는 개념은 동일하지만, policy를 따라간다는 것이 다르다.

#### action-value function

intput이 state만 들어가면 state-value function이고 state, action이 들어가면 action-value function 이다.

state s에서 action a 를 했을 때 그 이후에는 파이를 따라 에피소드를 끝까지 했을 때 나오는 return의 기댓값

ex\) Q-Learning, DQN 

### Bellman Expectation Equation

MDP를 풀기위한 방법

기대값에 대한 equation. 

### Optimal Value Function

#### optimal state-value function

\*는 optimal을 뜻한다. 

가능한 모든 policy 를 통해 나온 value function 중에서 가장 optimal 한 value function이 optimal value function 이다. 

#### optimal action-value function

가능한 모든 policy를 따른 q 함수 중의 max

#### 

#### Optimal Policy

두 개의 policy가 있을 때 어떤 policy가 나은지를 비교해야 한다. -&gt; partial ordering

모든 state들에 대해서 v 파이 프라임 보다 v 파이가 클 때, 파이가 파이프라임보다 크다고 할 수 있다.

* optimal policy 존재. 이 policy는 모든 state에 대해서 partial ordering이 성립한다.
* optimal policy를 따라가면 optimal value function이 된다. q도 마찬가지
* optimal policy는 하나가 아니라 여러개 일수도 있기 때문에 all optimal policies로 표시



MDP에서는 항상 deterministic optimal policy가 존재한다.

deterministic optimal policy? 기본적으로 policy는 stochastic 하지만 deterministic 하다는 것은 하나의 값만 1이고 나머지는 0인 것을 deterministic 하다고 한다. 항상 그렇게 된다는 것.

q 스타를 알고있으면 optimal policy를 알 수 있다.

### Bellman Optimality Equation

linear equation이 아니라서 bellman expectation처럼 식으로 전개해서 역행렬로 넘기고 수식적으로 풀 수가 없다. 

non-linear 하기 때문에 closed 해가 없다. 그래서 문제를 풀기 위한 iterative solution 들이 존재한다.

ex, value iteration, policy iteration, q-learning, sarsa

