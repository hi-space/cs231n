# 2. Markov Decision Processes

## Markov Processes

* MDP는 목표를 달성하기 위해 interactive 하게 학습하는 문제의 frame으로 고안됐다.
* MDP 는 fully observable 한 environment를 표현할 때 사용한다. 즉, state가 전체적인 process를 대표할 수 있어야 한다. 
* 대부분의 강화학습 문제는 MDP 형태로 만들 수 있다.
* MDP 구조는 목표 지향적인 상호작용으로부터 학습 문제를 추상화 한 것이다. 세부 장치가 무엇인지는 상관 없이 목표지향적인 행동을 학습하기 위해서는 agent-environment 사이에서 나오는 3가지 신호로 축약할 수 있다. \(state, action, reward\)

### Markov Property

{% hint style="info" %}
아래의 수식을 만족시키면 state $$S_t$$는 Markov 하다.

$$
\mathbb{P}[S_{t+1} | S_t] = \mathbb{P}[S_{t+1} | S_1, ..., S_t]
$$
{% endhint %}

위와 같은 수식을 만족시킬 때 state $$S_t$$는 `Markov` 하다고 할 수 있다. 바로 직전의 값만 있을 통해서만 현재 값을 정의할 수 있고, 그 이전의 값들의 사용하지 않는다는 것을 뜻한다.

![Markov Decision Process&#xC5D0;&#xC11C; Agent-Environment&#xC758; interaction](../.gitbook/assets/image%20%2887%29.png)

agent와 environment는 연속되는 discrete 시간 단계의 매 지점\(t = 0, 1, 2, 3, ..\)마다 interaction 한다. 모든 time step $$t$$에서 agent는 environment의 $$S_t$$를 받고, 그것을 기반으로 action$$A_t$$를 선택한다. 그리고 agent는 다음 time step에서 이전 action의 결과로 reward $$R_{t+1}$$를 받는다. 이 때 agent는 $$S_{t+1}$$에 있다고 인식한다.

MDP와 agent는 state, action, reward의 나열을 아래와 같이 만들어낸다. 이를 `history` 또는 `trajectory`라고 한다.

$$ S_0, A_0, R_1, S_1, R_2, A_2, R_3, ...$$

이 경우, 확률변수 $$R_t$$와 $$S_t$$는 바로 직전의 $$S_{t-1}$$과 $$A_{t-1}$$에만 의존된 이산 확률 분포를 갖는다. \($$R_t$$와 $$S_t$$는 Markov 하다\) 

state는 history 에서 뽑아내 만들 수 있고, history를 통해 state를 만든 후에는 history 정보들은 더이상 사용하지 않는다. 과거의 모든 agent-environment의 interaction은 미래에 변화를 가져오고, state는 과거의 모든 agent-environment 간 interaction에 대한 정보를 포함해야 한다. 이와 같은 조건을 만족하는 상태를 `Markov Property`를 가졌다고 표현한다.

$$
p(s', r|s, a) \doteq Pr\{S_t=s', R_t = r | S_{t-1} = s, A_{t-1} = a\}
$$

수식으로는 위와 같이 표현할 수 있다. 바로 직전의  $$S_{t-1}$$과 $$A_{t-1}$$에만 의거해 $$S_t$$와 $$R_t$$가 정해지고, 이를 MDP의 dynamics 로 정의한다. \($$p$$\) 

### State Transition Matrix

Markov Process에서는 n개의 state들이 discrete 하게 존재하고, 매 tick마다 state의 확률변수에 의해 state를 옮겨다닌다. 

$$ P{ss'} = \mathbb{P}[S_{t+1} = s' | S_t = s] $$

`state transition probability` 는 각 state들이 다음 state로 전이할 확률을 말한다. 수식으로 나타내면 위와 같다. $$S_t$$에서 $$S_{t+1}$$ 로 전이할 확률인 state transition probability를 $$P_{ss'}$$으로 나타낸다.

$$
P =  
\begin{bmatrix}
P_{11} & \cdots & P_{1n} \\
\vdots & \ddots & \vdots \\
P_{n1} & \cdots & P_{nn}
\end{bmatrix}
$$

`state transition matrix` 는 모든 state들이 다음 state로 전이할 확률\(state transition probability\)을 행렬로 표현한 것이다. 각 row 들의 합은 1이 된다.

### Markov Process \(Chain\)

Markov Process는 memoryless random process 이다. 즉, markov property 를 가진 state 들이 random 하게 sequence 를 이룬다는 말이다.

> * memoryless : 이전의 state를 보지 않는다. 
> * random process : 샘플링할 수 있다.

{% hint style="info" %}
Markov Process는 $$ <S, P> $$의 튜플이다.

* $$ S $$: 유한한 state 들의 set
* $$P$$: state transition probability matrix
  * $$P_{ss'} = \mathbb{P}[S_{t+1} = s' | S_t = s]$$
{% endhint %}

![Student Markov Chain](../.gitbook/assets/image%20%2810%29.png)

위의 그림은 Student 환경의 dynamics 에 대해 설명한 것이다. \(Markov Chain\) Markov Chain에서 episode를 sampling해서 표현할 수 있다. 각 episode 들은 특정 state에서 시작해서 final state에서 끝난다. 각 episode들은 독립적다.

* `episode` : 어느 state에서 시작해서 final state 까지 가는 것. episode를 sampling 한다고 한다.
* `final state (terminal state)` : 더이상 이동할 state가 없는 state

Chain이 $$S_1 = C1 $$ 에서 시작한다고 했을 때 아래와 같이 여러가지의 episode가 sampling 될 수 있다. 

* C1 C2 C3 Pass Sleep
* C1 FB FB C1 C2 Sleep
* C1 C2 C3 Pub C2 C3 Pass Sleep
* C1 FB FB C1 C2 C3 Pub C1 FB FB FB C1 C2 C3 Pub C2 Sleep

![transition graph / transition matrix](../.gitbook/assets/image%20%28124%29.png)

Markov chain은 위와 같이 transition graph로 표현할 수도 있고, state 간 transition matrix로 표현할 수도 있다.

## Markov Reward Process

{% hint style="info" %}
Markov Reward Process는 $$ <S, P, R, \gamma> $$의 튜플이다.

* $$ S $$: 유한한 state 들의 set
* $$P$$: state transition probability matrix
  * $$ P_{ss'} = \mathbb{P}[S_{t+1} = s' | S_t = s]$$
* $$R$$: reward function
  * $$R_s = \mathbb{E}[R_{t+1}|S_t = s]$$
* $$\gamma$$ : discount factor
  * $$ \gamma \in [0, 1]$$
{% endhint %}

![Student Markov Reward Process](../.gitbook/assets/image%20%28236%29.png)

Markov Reward Process는 Markov process에서 reward만 추가된 거다. 각 state 마다 reward를 가지고 있는다. \(확률적으로 옮겨지는 것이기 때문에 state에만 reward가 있으면 된다\)

### Return

{% hint style="info" %}
time step $$t$$때, total discounted reward인 return $$G_t$$는 아래와 같이 정의할 수 있다.

$$
\begin{matrix}
G_t &\doteq& R_{t+1} + \gamma R_{t+2} + ... 
\\ 
\\
&=& \sum_{k=0}^\infty \gamma^k R_{t+k+1}

\end{matrix}
$$
{% endhint %}

강화학습에서는 expected return을 최대화하고자 한다.

time step $$ t $$이후에 받는 연속된 reward를 $$ R_{t+1}, R_{t+2}, ...$$으로 나타낼 때, `return`$$G_t$$는 reward의 나열에 따른 어떤 특정 함수로 정의된다. 가장 간단하게 표현하면 reward들의 총합이다.

return은 reward와는 다른 개념이다. reward는 당장의 state에 대한 즉각적인 signal 값을 말하는 것이라면, return은 state들을 이동하면서 얻어진 reward들의 총합이다.

이 때, 단순히 reward를 더해주지 않는다. $$k+1$$ time step의 reward 값은 $$\gamma^kR$$ 이 된다. $$\gamma^k$$를 곱해줌으로써 미래의 reward는 그 값을 더 낮게해서 더해준다.

* $$\gamma$$가 0에 가까울 수록 : 근시안적인 evaluation을 하게 된다.
* $$\gamma$$가 1에 가까울 수록 : 미래지향적인 evaluation을 하게 된다.

#### discount를 하는 이유는? 

대부분의 Markov reward와 decision process는 discount 를 취해준다. 

* 수학적으로 편해서. discount 를 곱해야 수렴성이 커진다. 
* 사람/동물 행동은 보통 즉각적인 reward를 선호한다.
* 때때로 undiscounted Markov reward processes \($$\gamma=1$$\) 가 있을 수 있는 데, 이 경우는 모든 sequence가 무조건 terminate 된다는 것이 보장되어야 한다.

### Value Function

{% hint style="info" %}
`state value function` $$v(s)$$는 state $$s$$에서 시작했을 때 얻어지는 expected return \(return의 기대값\) 이다.

$$
v(s) = \mathbb{E}[G_t|S_t = s]
$$
{% endhint %}

특정 state에 왔을 때 여러개의 sampling 된 episode가 존재할 거고, 각 episode 마다 return 값이 있을 것이다. 그 return 값들을 평균낸 값이 $$v(s)$$가 된다. 즉, $$G_t$$는 확률변수이다.

![Sample returns for Student MRP](../.gitbook/assets/image%20%2844%29.png)

$$S_1 = C1 , \gamma={1\over2}$$ 일 때, 각 episode 마다 return 값을 얻을 수 있다. 

### Bellman Equation for MRPs

> **Bellman Equation** : 어떤 상태의 값과 그 후에 이어지는 상태의 값들 사이의 관계를 표현하는 방정식

value function은 bellman equation에 의해 iterative 하게 계산된다.

$$
\begin{matrix}
G_t &\doteq& R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \cdots \\ &=& R_{t+1} +  \gamma (R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \cdots)
\\ &=& R_{t+1} + \gamma G_{t+1}
\end{matrix}
$$

![](../.gitbook/assets/image%20%28215%29.png)

value function은 결과적으로 두가지 파트로 나눠서 표현할 수 있다.  $$\gamma$$로 묶고 $$G_{t+1}$$로 치환하고 $$v(S_{t+1})$$로 치환하는 단순한 점화식 문제로 표현하면, value function이 결과적으로 아래와 같은 두가지 파트로 나눠 표현할 수 있다는 것을 알 수 있다.

* $$R_{t+1}$$: 즉각적인 reward
* $$\gamma v(S_{t+1})$$: 다음 state의 discounted value

위와 같이 $$S_{t+1}$$과 $$S_t$$의 value function 사이의 관계를 식으로 나타낸 것을 Bellman Equation 이라고 한다.

![](../.gitbook/assets/image%20%2884%29.png)

Bellman Equation을 Matrix로 표현하면 아래와 같다.

![](../.gitbook/assets/image%20%281%29.png)

![](../.gitbook/assets/image%20%2827%29.png)

이것은 Bellman Equation을 도식화 한 것이다. 

* ○ : state
* ● : state-action pair

1. agent는 최상위의 root node인 상태 $$s$$에서 시작한다.
2. agent는 policy $$\pi$$에 의해 3가지 중 하나의 action $$a$$을 선택한다. 
3. 다음 state $$s'$$이 reward $$r$$와 함께 얻어진다.
4. 이 과정에서 $$p$$로 부터 environment의 dynamics 가 주어진다.
5. Bellman Equation은 모든 가능성에 해 그것이 발생할 확률을 가중치로하여 평균값을 계산한다.

## Markov Decision Process

MDP에는 Markov Reward Process에서 action이 추가된 것이다. 모든 state가 Markov 한 환경을 말한다.

{% hint style="info" %}
Markov Reward Process는 $$ <S, A,  P, R, \gamma> $$의 튜플이다.

* $$ S $$: 유한한 state 들의 set
* $$A$$: 유한한 action 들의 set
* $$P$$: state transition probability matrix
  * $$ P_{ss'}^a = \mathbb{P}[S_{t+1} = s' | S_t = s, A_t=a]$$
* $$R$$: reward function
  * $$R_s = \mathbb{E}[R_{t+1}^a|S_t = s, A_t=a]$$
* $$\gamma$$ : discount factor
  * $$ \gamma \in [0, 1]$$
{% endhint %}



![Student MDP](../.gitbook/assets/image%20%28320%29.png)

MRP에서는 state 에 reward가 있었는데 MDP 에서는 action 마다 reward가 주어진다. 특정 action이 항상 같은 state로 가지않고, state의 전이 확률에 따라서 확률적으로 어떤 state로 갈 지 정해진다. 

### Policy

MRP에서는 state간 이동을 할 때 확률분포에 의해서만 이동했다. 하지만 MDP에서는 action에 대한 확률에 따라 state로 이동하게 된다. agent의 행동을 결정하기 위해 특정 action을 수행하기 위한 policy가 필요하다. 

{% hint style="info" %}
policy $$\pi$$는 주어진 state에서 어떤 action을 수행할 것인지 나타내는 distribution이다.

$$
\pi(a|s) = \mathbb{P}[A_t=a|S_t=s]
$$
{% endhint %}

* policy는 agent의 행동들을 완전히 정의한다.
* MDP policy는 오직 현재의 state에만 의존해 결정된다. \(과거의 state들은 무시한다\)
* 즉, policy는 stationary \(time-independent\) 하다고 말할 수 있다.

어떤 MDP $$M=<S, A, P, R, \gamma>$$가 있고 policy $$\pi$$가 고정되어 있다고 하자. 이 때 아래가 정의가 성립된다.

* state sequence \($$S_1, S_2, ...$$\)는 Markov process 가 다. \($$S, P_{\pi}$$\)
* $$P_{s,s'}^{\pi} = \sum_{a \in A} \pi(a|s)P_{ss'}^a$$

policy $$\pi$$가 정해져있으면 다음 state로 넘어갈 수 있는 state transition을 미리 다 계산해 줄 수 있기 때문에 Markov Process라고 할 수 있다.

policy가 $$\pi$$일 때 $$s$$에서 $$s'$$으로 갈 확률은, $$s$$에서 action $$a$$를 선택할 확률과 action $$a$$를 했을 때 $$s$$에서 $$s'$$으로 갈 확률을 다 더하면 된다.

* state, reward sequence \($$S_1, R_1, S_2, ...$$\)는 Markov Reward Process 가 된다. \($$S, P^{\pi}, R^{\pi}, \gamma$$\)
* $$R_s^{\pi} = \sum_{a \in A} \pi(a|s)R_s^a$$

reward도 위와 같은 개념으로 미리 계산해 줄 수 있기 때문에 state와 reward가 Markov인 Markov Reward Process가 된다고 할 수 있다.

policy가 $$\pi$$일 때 $$s$$의 reward는 s에서 $$a$$를 선택할 확률과 $$a$$일 때 $$s$$의 reward 를 다 더해주면된다.

### Value Function

MRP에서는 action이 없었기 때문에 policy가 필요 없었지만, MDP에서는 action이 추가되었기 때문에 policy가 필요하다. MDP의 value function에는 state-value function과 action-value function 이 있다.

#### state-value function

{% hint style="info" %}
state-value function $$v_{\pi}(s)$$는 policy $$\pi$$를 따랐을 때, state $$s$$에서 시작해서 얻게 되는 return의 기댓값\(expected return\) 이다.

$$
v_{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s]
$$
{% endhint %}

> \*\*\*\*$$\mathbb{E}_{\pi}$$: agent가 policy $$\pi$$를 따랐을 때 어떤 확률변수의 기대값

state $$s$$에서 시작한 이후로 policy $$\pi$$를 따랐을 경우 dㅓㄷ게 되는 return의 기댓값이다. episode를 sampling해서 나오는 return 값의 기댓값이라는 개념은 동일하지만, policy를 따라간다는 것이 MRP의 value function과 다르다.

#### action-value function

{% hint style="info" %}
action-value function $$q_{\pi}(s, a)$$는 policy $$\pi$$를 따라 state $$s$$에서  action $$a$$를 선택해 시작하고 얻게 되는 return의 기댓값\(expected return\) 이다.

$$
q_{\pi}(s,a) = \mathbb{E}_{\pi}[G_t|S_t  = s, A_t=a]
$$
{% endhint %}

> $$q_{\pi}(s, a)$$ : policy $$\pi$$를 따를 때 state $$s$$에서 action $$a$$를 취하는 것의 가치

state $$s$$에서 action $$a$$를 취하고 그 이후에는 policy $$\pi$$를 따를 경우 얻게 되는 return의 기댓값이다. Q-Learning, DQN 등이 이 방식을 따른다.

정리하면, input이 state만 들어가면 state-value function이고 state, action이 들어가면 action-value function 이다.

value function $$v_{\pi}$$, $$q_{\pi}$$는 경험으로부터 추정 가능하다. agent가 policy $$\pi$$를 따라서 마주치는 state 들의 평균들을 나열하다보면 state 갯수가 무한으로 갈 때 $$v_{\pi}$$로 수렴할 것이다. 이와 마찬가지로 각 state에서 얻어지는 각 action 값의 평균들은 $$q_{\pi}$$로 수렴할 것이다. 이것이 바로 Monte Carlo Methods 인데, random sample된 여러개의 실제 return 값을 통해 평균을 계산하기 때문이다.

### Bellman Expectation Equation

MDP를 풀기위한 방법으로 기댓값에 대한 equation이다.

#### state-value function

{% hint style="info" %}
state-value function을 Bellman equation으로 표현한 것이다.

$$
v_{\pi}(s) = \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1})|S_t = s]
$$
{% endhint %}

![](../.gitbook/assets/image%20%2818%29.png)

이는 state와 state-action과의 관계를 backup diagram으로 도식화 한 것이다. 

* ○ : state
* ● : state에서 action을 선택한 상황

○ state에서는 ● 개의 갯수 만큼 action을 선택할 수 있다. 이 때 $$v_{\pi}(s)$$와 $$q_{\pi}(s,a)$$의 관계는 위의 식처럼 표현할  있다. 

각 action을 할 확률과 그 action을 해서 받는 expected return을 곱하면 현재 state의 value function이 된다.

#### action-value function

{% hint style="info" %}
action-value function을 Bellman equation으로 표현한 것이다.

$$
q_{\pi}(s, a) = \mathbb{E}_{\pi}[R_{t+1} + \gamma q_{\pi}(S_{t+1}, A_{t+1})|S_t = s, A_t =a]
$$
{% endhint %}

![](../.gitbook/assets/image%20%2892%29.png)

state-value function의 그림에서 reward $$r$$이 추가된거다. state와 action을 받아 reward가 나온다. 다음 state $$s'$$은 deterministic 하면 하나의 ○ 로만 표현되겠지만, stochastic한 환경에서는 여러개의 state들이 확률\(state transition probability matrix\)로 나올 수 있다.

action-value function은 immediate reward + \(action을 취해 각 state로 갈 확률 \* 그 위치에서의 value function\) 이다.

![](../.gitbook/assets/image%20%28246%29.png)

state-value function과 action-value function 도표를 통합하여 표현하면 위와 같이 나타낼 수 있다.

Bellman Equation에서 value function $$v_{\pi}$$는 유일한 해고, reward와 sate transition probability는 미리 알 수 없다. 수많은 시도를 통해 경험적으로 알게되고 모든 정보를 알게되면 그것이 MDP의 model이 된다. 하지만 실제 강화학습 문제에서는 모든 정보를 알아내기 어렵기 때문에 Bellman Equation으로는 구할 수 없다. Bellman Equation은 Dynamic Programming과 같이 discrete한 time 에서의 최적화 문제에서 적용할 수 있다.

### Optimal Value Function

#### optimal state-value function

{% hint style="info" %}
**optimal state-value function** $$ v_*(s)$$는 모든 policy 에서 가장 maximum 한 value function이다.

$$
v_*(s) = \max_{\pi} v_{\pi}(s)
$$
{% endhint %}

> **\*** :  optimal

가능한 모든 policy 를 통해 나온 value function 중에서 가장 optimal 한 value function이 optimal value function 이다. 

#### optimal action-value function

{% hint style="info" %}
**optimal action-value function** $$ q_*(s,a)$$는 모든 policy 중에서 가장 maximum한 action-value function이다.

$$
q_*(s,a) = \max_{\pi}q_{\pi}(s,a)
$$
{% endhint %}

가능한 모든 policy 를 통해 나온 action-value function 중에서 가장 optimal 한 action-value function이 optimal action-value function 이다.

가장 optimal 한 value function을 찾았을 때 MDP 문제가 풀렸다고 말한다. 

#### Optimal Policy

두 개의 policy가 있을 때 어떤 policy가 나은지를 비교해야 한다.

$$ \pi \ge \pi' \quad if \; v_{\pi}(s) \ge v_{\pi'}(s), \forall s$$

모든 state들에 대해서 $$  v_{\pi}(s) \ge v_{\pi'}(s)$$일 때, $$  \pi \ge \pi'$$라고 할 수 있다. \(`partial ordering`\) 즉, 다른 모든 policy 보다 좋거나 같은 policy가 항상 하나 이상 있다. 이것을 optimal policy라고 하고 $$ {\pi}_*$$로 표현한다.

MDP 에서는 아래와 같은 정의가 성립된다.

* 모든 state에 대해서 partial ordering이 성립하는 deterministic optimal policy가 존재한다.
* optimal policy를 알고 있다면 optimal value function, optimal action-value function 도 알 수 있다. $$v_{\pi_*}(s) = v_*(s) \quad , \quad q_{\pi_*}(s, a) = q_*(s, a)$$

![](../.gitbook/assets/image%20%28230%29.png)

optimal action-value function를 찾으면 optimal policy를 알 수 있다. optimal action-value function를 안다면 q 값이 높은 action만 선택하면 되기 때문에 MDP 문제가가 풀린 것이다. 가장 높은 action 만 선택하기 때문에 deterministic 하다.

> **deterministic optimal policy** : 기본적으로 policy는 stochastic 하다. policy가 deterministic 하다는 것은 하나의 값만 1이고 나머지는 0인 것을 deterministic 하다고 한다. 항상 그렇게 된다는 것.

### Bellman Optimality Equation

optimal policy를 구하기 위한 Bellman Equation을 Bellman Optimality Equation 이라고 한다. 

Optimal bellman equation은 각 state 마다 하나씩의 equation으로 구성되어 있다. 즉, n 개의 state가 있으면 n개의 미지수에 대한 n개의 방정식이 존재하는 거다.

![Bellman Optimality Equation for optimal value function](../.gitbook/assets/image%20%28179%29.png)

optimal policy를 따르는 어떤 state의 value function은 그 state에서 선택할 수 있는 best action 의 expected return 값과 같아야 한다.

![Bellman Optimality Equation for optimal action-value function](../.gitbook/assets/image%20%2898%29.png)

#### Optimal Value Function

optimal value function\($$ v_*$$\)을 구하면 optimal policy를 결정하는 것은 쉽다. 각 state $$s$$에 대해 optimal bellman equation의 value function이 최댓값이 되도록하는 action이 하나 이상 있을 것이다. 이러한 action에 대해서 0이 아닌 확률을 부여하는 policy는 그 무엇든 optimal policy가 될 수 있다. optimal value function을 얻고 나면 optimal value function에 대한 greedy 한 policy가 optimal policy라는 것이다.

greedy 한 선택은 일반적으로 장기적으로 더 좋은 대안을 선택할 수 있는 기회를 선택하지 않고 당장의 최적의 선택을 하는 것을 뜻한다. 하지만 optimal value function을 통한 greedy 한 선택은 장기적인 측면에서 최적의 결과를 가져온다. optimal value function에는 미래에 일어날 수 있는 모든 action에 대한 reward의 결과가 담겨있기 때문이다.

#### Optimal Action-Value Function

optimal action-value function이 optimal policy를 만드는 것은 훨씬 쉽다. $$ q_*$$ 만 있으면 agent는 위와 같은 탐색을 하지 않고도 모든 state에 대해 $$q_*(s,a)$$를 최대로 만드는 행동을 간단하게 찾을 수 있다. 

optimal action-value function은 다음 state와 그에 대한 value, 환경의 dynamics 에 대한 정보 없이도 optimal 한 action을 선택할 수 있도록 해준다.

![](../.gitbook/assets/image%20%2890%29.png)

Bellman Expectation의 backup diagram과 비슷해 보이지만 한가지 다른 것은 max 값이 표시되어 있다는 점이다. 주어진 policy에 대해 value의 기대값이 아닌 최댓값이 적용되었음을 나타내기 위해 arch 표시가 추가됐다.

![](../.gitbook/assets/image%20%28332%29.png)

![](../.gitbook/assets/image%20%2895%29.png)

linear equation이 아니라서 Bellman Expectation처럼 식으로 전개해서 역행렬로 넘기고 수식적으로 풀 수가 없다. Non-linear 하기 때문에 closed 해가 없다. 그래서 문제를 풀기 위한 iterative solution 들이 존재한다. \(ex, value iteration, policy iteration, Q-learning, SARSA\)

Bellman Equation을 통해 iterative 하게 MDP 문제를 푸는 것을 Dynamic Programming 이라고 한다.

## Reference

{% embed url="https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf" %}



