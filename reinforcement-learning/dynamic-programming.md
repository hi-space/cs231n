# Planning by Dynamic Programming

## Introduction

* Planning : environment의 model을 알고 문제를 푸는 것 \(model-based\)
* Learning : environment의 model을 모르고 상호작용을 통해 문제를 푸는 것 \(model-free\)

Dynamic Programming은 Planning 으로, environment의 model을 알고 있다는 가정하에 Bellman Equation으로 문제를 푸는 방식이다. 완벽한 모델과 엄청난 양의 계산이 필요하기 때문에 실제 강화하기 문제에서는 적용하기 어려울 수 있다.

### What is Dynamic Programming

DP는 ‌MDP와 같은 environment model이 완벽하게 주어졌을 때 optimal policy를 계산하기 위해 사용 되는 하나의 알고리즘이다.

DP로 문제를 풀기 위해서는 하나의 복잡하고 큰 문제를 sub-problem 으로 나누고, sub-problem에 대해 문제를 푼 후 그 해를 combine 하는 형식으로 진행된다.

### Requirements

DP로 문제를 풀기 위해서는 아래와 같이 두가지 속성이 있다.

* Optimal substructure : optimal solution은 sub-problem으로 나눠질 수 있다.
* Overlapping sub-problems : 다수의 sub-problem이 만들어질텐데, 이 각각의 sub-problem의 solution은 cache 로 저장해놨다가 나중에 재사용이 가능해야 한다.

MDP는 위와 같은 두가지 특성을 모두 만족시킨다. Bellman Equation이 recursive 하게 표현되기 때문에 sub-problem으로 나뉘어 질 수 있고, value function을 저장해뒀다가 solution에 재사용할 수 있다. 그래서 MDP 환경은 Dynamic Programming을 사용하기 적합하다.

### Planning by Dynamic Programming

Dynamic Programming은 MDP 환경의 모든 것을 알고 있어야 한다. 이는 MDP의 planning 문제에 사용되는 방법이다. Planning 을 풀기 위해서는 prediction, control 두가지 스텝으로 나뉜다.

Optimal 하지 않은 policy에서 value function을 구하고\(prediction\) 현재의 value function을 토대로 더 나은 policy를 구하는 과정을 반복하며 optimal policy를 구한다.\(control\)

#### Prediction

* **Input** : $$ MDP <S, A, P, R, \gamma>,  \;\pi $$   or  $$MRP <S, P^{\pi}, R^{\pi}, \gamma>$$
* **Output** : value function $$v_{\pi}$$
* MDP와 policy가 주어졌을 때 value function이 어떻게 되는지 찾는 문제
* policy evaluation 

#### Control

* **Input** : $$ MDP <S, A, P, R, \gamma>$$
* **Output** : optimal value function $$v_*$$ ,  optimal policy $$ {\pi}_*$$
* policy 없이 MDP만 있을 때 optimal policy와 optimal value function을 찾는 문제
* policy iteration, value iteration 

## Policy Evaluation

### Iterative Policy Evaluation

{% hint style="info" %}
**\[Problem\]** 주어진 어떤  policy를 평가  
**\[Solution\]** Bellman Expectation Backup으로 iterative 하게 문제를 풀어나간다.  
**\[Result\]**     value function 이 구해짐
{% endhint %}

Policy Evaluation은 Prediction 문제를 푸는 것으로, 주어진 policy에 대한 evaluate 을 통해 Bellman Equation을 반복적으로 수행하며 value function을 구해낸다.

> **synchronous backup** : 모든 state 들에 대해 parallel 하게 동시에 한번씩 업데이트

synchronous backup으로 $$ v_1 \rightarrow v_2  \rightarrow v_3... \rightarrow   v_{\pi}$$ 를 one step 씩 업데이트 한다. 반복하다 보면 $$ v_{\pi} $$ 로 수렴하게 되는 순간이 온다.

각 iteration $$k+1$$에서 모든 state 들 대상으로 $$v_k(s')$$ 에서 $$v_{k+1}(s)$$로 업데이트 한다. 



![](../.gitbook/assets/image%20%2867%29.png)



![](../.gitbook/assets/image%20%28275%29.png)

아래와 같이 MDP 모델을 정의한다고 하자.

* normal state 14개
* terminal state 1개
* action 4개 \(random policy\)
* undiscounted episodic MDP \($$\gamma = 1$$\) 
* time step 마다 reward = -1

이 때 agent의 uniform random policy는 이와 같이 정의 할 수 있다. 

$$ \pi(n|\cdot) =  \pi(e|\cdot) =  \pi(s|\cdot) =  \pi(w|\cdot) = 0.25$$

evaluation은 이 policy가 얼마나 좋은지 평가하는 것이고, 그것은 그 policy를 따라 갔을 때 받게 되는 value function 을 통해 알 수 있다. 

![](../.gitbook/assets/image%20%28278%29.png)

처음에는 4방향 모두 갈 수 있는 random policy로 시작한다. value function은 Bellman equation을 통해 구할 수 있기 때문에 k가 증가함에 따라 one step 씩 각 state의 value function을 업데이트 할 수 있다. 이 때 사용되는 Bellman Equation은 위에서 나왔듯이 아래와 같다.

$$
v_{k+1}(s) = \sum_{a \in A} \pi(a|s)(R_s^a + \gamma \sum_{s' \in S}P_{ss'}^av_k(s')
$$

random한 policy를 무한대까지 evaluation 하고 평가된 value에서 greedy하게 움직이다보면 optimal policy를 찾아낼 수 있다. \(Policy Iteration\)

## Policy Iteration‌

### How to Improve a Policy

![](../.gitbook/assets/image%20%28283%29.png)

이제 optimal policy를 찾기 위해 policy를 더 나은 policy로 update 해줘야 한다. 그래서 value function을 찾기 위한 policy가 있고, 해당 value function에 의해 greedy 하게 움직이는 policy 두가지가 있다. 

* **Evaluate the policy** $$\pi$$ : value function을 찾기 위한 policy $$  v_{\pi}(s) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + .. | S_t = s]$$
* **Improve the policy** $$\pi'$$: 평가하는 value function에 의해 greedy 하게 움직이는 policy $$ \pi' = greedy(v_{\pi})$$

결국 $$\pi' = \pi*$$ 가 될 때 까지 evaluation과 improvement를 반복하다보면 항상 optimal policy $$ \pi*$$로 converge 한다.

### Policy Improvement

policy improve 를 해서 greedy 하게 action을 취하면 정말 개선될까?

s 에 있을 때 파이를 따라서 value function을 따른 값 = s에서 파이가 골라준 action을 해서 value를 따라가는 값

max 어쩌구 → greedy 한 선택

bellman equation으로 푼거

언젠까는 더이상 improve되지 않는 상황이 온다

어떤 알고리즘을 써도 evaluaitnon이 되고 imporve 가 된다.

## Value Iteration‌

### Principle of Optimality

optimal policy는 두가지 컴포넌트로 나뉜다.

* 첫번째 action 이 optimal 해야 한다.
* 그 이후 state 에서 부터는 optimal policy를 따른다.

### Deterministic Value Iteration

한 sub-problem의 solution인 $$ v_*(s')$$를 알 수 있다면, $$ v_*(s)$$도 아래와 같은 식을 이용해 알 수 있다.

$$
v_*(s) \leftarrow \max_{a \in A}R_s^a + \gamma \sum_{s' \in S} P_{ss'}^av_*(s')
$$

final reward에서 시작하게 되면 goal에 도달하기 직전 step 들이 있을 것이고, 그 이전으로 돌아가며 iterative 하게 연계산을 한다.

iterative 하게 bellman optimality equation을 하다보면

goal에 도달하기 위핸 직전 스텝들이 있고, 그 이전으로 이전으로 가면서 역계산 하는 것.

이번엔 policy가 없다. value 만 가지고 optimal value function을 구한다

### Value Iteration

optimal policy를 찾는 문제

Bellman optimality backup을 iterative 하게 적용해서 푼다

policy iteration과 다르게 policy 가 없다

synchronous : 한타임에 한꺼번에 업데이트 하는 방식

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2e2d2fac-2d63-4c34-a00a-f7db5044fbaf/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2e2d2fac-2d63-4c34-a00a-f7db5044fbaf/Untitled.png)

state-value function이라면 굉장히 복잡하다. O\(mn^2\)

모든 state들을 돌아가면서 한번씩 다 업데이트 해야 하니까.

비효율이 많기 때문에 개선해야 한다. 다음 단원

## Extensions to Dynamic Programming

DP는 synchronous bakups을 이용했다. parallel 하게 동시에 업데이트 하는 방식.

asynchrounous 하게 하면 computation power를 줄일 수 있고 converge 한 것이 보장된다\(골고로 state들을 다 업데이터 해줘야 함\)

### In-Place Dynamic Programming

value function을 한 테이블만 가지고 있고, 한 array만 업데이트 하는 방식

### Prioritised Sweeping

bellman error가 컸던 애들이 중요한 애들이다. table의 값 차이가 큰 애들. 걔네들을 먼저 업데이트 한다.

### Real-time Dynamic Programming

agent가 방문한 state 들을 먼저 업데이트 한다.

### Full-Width Backups

DP는 full width backups을 사용한다. 하지만 애초 큰 문제에서는 이렇게 할 수가 없다. 차원의 저주에 빠진다. state갯수가 늘어날 수록 exponential \(발산\) 한다.

그래서 sample backups 한다.

### Sample Backups

state가 많아도 고정된 cost로 bakcup할 수 있다. model-free에서도 사용할 수 있다.

어디에 도착할 지 몰라도 sample을 뽑아서 그걸로 backup 한다.

