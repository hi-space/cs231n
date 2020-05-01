# Introduction

![](../.gitbook/assets/image%20%28316%29.png)

강화학습 : 주어진 상황에서 어떠한 행동을 취할지를 학습하는 것

강화학습은 Supervised Learning, Unsupervised Learning 과 함께 Machine Learning의 한 분야이다.

* Supervised Learning은 label이 포함되어 있는 데이터들을 통해서 학습하는 것으로, unseen data에 대해서도 예측할 수 있도록 하는 것이 목표이다. 이 방식은 상호작용이 필요한 문제에는 적합하지 않다. 상호작용 문제에서는 경험의 의거하여 바람직한 행동을 선택해야하기 때문에, 모델은 새로운 환경이 주어졌을 때 미지의 영역에서의 자신의 경험에 의거하여 배울 수 있어야 한다. 
* Unsupervised Learning은 label이 없는 데이터의 집합 안에서 숨겨진 구조를 찾아낸다. 강화학습은 보상을 최대로 하기 위해 노력하지만 숨겨진 구조를 찾으려고 하지 않는다. 구조를 찾아내는 것이 어느정도 강화학습에 도움이 될 수는 있지만, 구조를 파악한다고 항상 최대의 보상을 얻는 선택을 할 수 있는 것은 아니다.

다른 learning 과 달리 강화학습에서는 exploration과 exploitation 사이의 trade-off 를 하는 것이 중요한 문제 중 하나이다. 이미 경험한 행동들을 활용\(exploitation\) 해서 큰 보상을 주는 행동을 취해야 하지만 동시에 더 좋은 행동을 선택하기 위해 탐험\(exploration\) 해야 한다. 

강화학습의 또 하나의 key feature는, 미지의 환경과 interaction 하는 agent 를 학습할 때, 필요한 모든 sub-problem들을 전체적으로 고려한다는 것이다. 하나의 큰 문제를 풀기 위해선 여러가지 풀어야 하는 작은 문제들이 있을 수 있는데 이것들을 포괄하여 학습하는 것이다. goal-seeking agent를 명확하게 정의하고 환경과의 interaction을  통해 planning과 control이 실시간으로 작용되도록 해야한다.

예를 들어 실내 주행하는 로봇을 만든다고 했을 때, 카메라를 통해 전방의 물체를 인식하고 판단하거나 원하는 목적지까지 주행하기 위해 바퀴의 토크값을 얼마나 줄 것인지 등의 하위 문제들이 있을 수 있다. 여기에서 가장 큰 문제는 목적지까지 로봇이 정상적으로 주행하는 것이지만, 그 문제를 해결하기 이전에 인식의 정확도를 높이거나 정밀한 컨트롤을 하기 위해 연구가 필요할 수 있다. 강화학습은 위와 다른 접근법으로, 이러한 하위 문제들을 완벽하게 planning 하지 않고도 환경과 상호작용하며 적절히 주행할 수 있는 방법을 학습하려고 노력할 것이다.

강화학습으로 푸는 문제들은 학습하려고 하는 학습자와 그를 둘러싼 주변 환경 사이의 interaction 이 있다. 주변 환경에는 uncertainty 요소들이 있지만, 학습자는 goal을 이루기 위한 방법들을 모색한다. 당장 주어진 상황에서 어떻게 선택을 해야 하는지 판단하고, 그 판단으로 인해 미래의 환경이 어떻게 달라져서 학습자에게 어떻게 영향이 오는지 까지 고려해서 선택을 계획해야 한다.

* Trial and Error
* Optimal Control

## Characteristics of Reinforcement Learning

* supervisor가 없고 reward signal 만 존재한다. 
* Feedback이 즉각적이지 않다. 여러가지의 행동을 취한 뒤 지연된 피드백을 받게 될 수 있다.
* 시간 독립적이지 않고 sequential 하다.
* Agent의 action이 이후의 상태 데이터에 영향을 미친다.

## Elements of Reinforcement Learning

학습자와 환경을 제외하고도 네가지 주요한 구성요소가 있다. `policy`, `reward signal`, `value function`, 환경에 대한 `model` \(optional\)

### 정책 \(Policy\)

특정 시점에 학습자가 취하는 행동을 정의. 학습자가 환경을 보고 어떤 행동을 취해야 하는지 알려주는 것으로, 정책을 통해 학습자의 행동을 결정할 수 있다. 

정책은 간단한 함수, look up table 이 될 수도 있고, 각 행동이 선택될 확률을 부여하고 그 확률에 따라 행동을 선택할 수도 있다\(확률론적으로 선택\)

### 보상 신호 \(Reward signal\)

강화학습이 성취해야 할 목표를 정의. 매 시간마다 환경은 학습자에게 reward라고 하는 하나의 숫자 signal을 전달한다. 학습자는 reward signal을 통해 자신의 취한 행동이 좋은 것인지 나쁜 것인지 판단할 수 있다.

정책이 선택한 어떤 행동을 해서 적은 reward signal을 받게되면, 다음번 유사한 환경에서는 정책이 바뀌어 다른 선택을 하도록 유도할 수 있다. 

일반적으로 reward signal은 환경의 상태와 취해진 행동에 대해 확률적으로 결정되는 stochastic 함수가 될 수 있다.

### 가치 함수 \(value function\)

reward signal 은 어떤 행동에 대해 즉각적으로 알려주는 반면, value function은 장기적인 관점에서 무엇이 좋은 선택인지 알려준다.

특정 상태의 value는 특정 시점 이후부터 일정 시간 동안 학습자가 기대할 수 있는 reward의 총량이다. 일정 시간 동안의 상태를 고려해 long-term으로 평가한 지표이다.

value는 reward에 대한 예측으로, value가 필요한 것은 더 많은 reward를 얻기 위함이다. 행동을 선택함에 있어서 가장 중요한 것은 reward 지만, reward가 최대인 행동을 하기 위해선 당장의 reward가 높은 행동을 취하기 보다는 value가 최대인 값을 선택해야만 장기적으로 최대한 많은 reward를 얻을 수 있다. 

reward는 당장 환경에서 나오는 데이터들을 토대로 쉽게 얻을 수 있지만 value를 정의하는 것은 그보다 어려운 일이다. 학습자의 관측값을 통해 반복적으로 추정되어야 한다. 이 부분이 강화학습에서 가장 핵심적인 역할을 한다.

### 환경 모델 \(Model\)

Environment Model은 환경의 변화를 모사한다. 현재 상태와 그에 따라 취해지는 행동으로부터 다음 상태와 보상을 예측한다. 환경이 어떻게 학습자의 행동에 반응할 것인지를 예측하기 위해 만든 것으로 imagination\(simulation\) 하기 위함이다.

실제 상황을 경험하기 이전에 일련의 행동을 결정\(planning\)하기 위해 사용된다.  Planning은 Model을 통해 imagination\(simulation\) 하여 어떠한 policy를 만들고 향상시키는 과정을 뜻한다.

* `model-based` : model과 planning을 사용하여 강화학습 문제를 해결하는 방법. agent가 model을 가지고 있다.  \(ex, Q-Planning, Rollout Algorithms, Monte-Carlo Tree Search, SARSA, Actor-Critic\) 
* `model-free` : 시행착오를 통해 환경 모델을 학습하고 동시에 그 모델을 사용하여 계획하는 방법 \(ex, Q-Learning, DQN, A3C\)

Model-based는 planning에, Model-free는 learning에 중점을 두고 있다.

## The RL Problem

## \# Reward

$$ R_t $$: $$t$$ 번째 시간에 agent가 얼마나 잘했는지 알려주는 scalar 피드백 시그널 

* reward를 최대화 하는 것이 agent의 목표이다.
* RL은 reward hypothesis를 기반으로 한다.

{% hint style="info" %}
**Reward Hypothesis :** All goals can be described by the maximisation of expected cumulative reward \(축적된 reward를 극대화\)
{% endhint %}

Sequential Decision Making 해야 한다. 연속적으로 선택을 해가면서 reward의 총합을 최대화 하는 것이 목표이기 때문에 long-term으로 생각해야 한다. 

## Agent and Environment

![](../.gitbook/assets/image%20%28122%29.png)

Agent가 어떤 Action을 하게 되면 Environment는 그 행동에 대해 달라진  observation과 reward를 반환해준다.

매 time step 마다 agent는 

* $$A_t$$: Executes action
* $$O_t$$: Receives observation
* $$R_t$$: Receives scalar reward

Environment는 

* $$A_t$$: Receives
* $$O_{t+1}$$: Emits observation
* $$R_{t+1}$$: Emits scalar reward

## History and State



