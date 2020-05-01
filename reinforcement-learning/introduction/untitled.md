# Multi-Armed bandit problem

## A k-armed Bandit Problem

다음과 같은 문제가 있다.

![Multi-Armed Bandits](../../.gitbook/assets/image%20%2838%29.png)

외팔이 강도\(one-armed bandit\)는 앞의 여러대\(`k`\)의 슬롯머신 레버\(`arm`\)를 당겨, 그에 해당하는 보상\(`reward`\)을 얻을 수 있다. 단, 한번에 한개의 레버만 당길 수 있고 각 슬롯머신의 보상은 다르다. 이 보상은 선택된 행동에 따라 결정되는 고정 확률 분포\(`stationary probability distribution`\)로부터 얻어진다. 이 문제에서 주어진 시간 \(`time step`\) 동안 보상을 최대화 할 수 있는 정책\(`policy`\)을 배우고자 한다.

k-armed bandit problem에서 k개의 행동 각각에는 그 행동이 선택됐을 때 평균 보상값\(mean reward\)이 얻어진다. 이를 k action에 대한 `value` 라고 한다. 

$$
q_*(a) \doteq \mathbb{E}[R_t | A_t = a]
$$

* $$A_t $$        :  시간 단계 $$t$$에서 선택되는 action
* $$R_t$$        :  위의 action을 취했을 때 얻어지는 reward
* $$ q_*(a)$$  :  행동 $$a$$가 선택됐을 때 얻어지는 reward의 expectation
* $$Q_t(a)$$ :  시간 $$t$$에서 추정한 행동 $$a$$의 reward

모든 action에 대한 reward를 알고 있다면 항상 높은 reward를 주는 action만을 선택하면 되기 때문에 k-armed problem 문제는 쉽다. 하지만 보통 action에 대한 reward를 추정할 수는 있으나 정확하게 알 수는 없다. 그래서$$Q_t(a)$$ 가 기대값 $$ q_*(a)$$와 가까워질 수록 정확한 추정이 될 수 있다. 이렇게 value를 추정할 수 있다면 각 시간 $$t$$마다 value가 최대인 action을 결정 할 수 있다. 

최대의 value 값을 선택하는 행동을 greedy 한 행동이라고 할 수 있다. greedy action을 선택하는 것은 현재까지 갖고 있는 지식을 exploiting 하는 것이다. 그와 반대로 현재까지 가지고 있는 지식을 활용하지 않고 non-greedy 한 action을 선택하는 것을 exploring 이라고 한다. 당장의 action에 대해 최대의 reward를 얻고자 하면 greedy한 action을 선택하는 것이 맞지만, 장기적으로 봤을 때 reward의 총합을 최대화 하기 위해선 exploration이 더 좋은 선택일 수 있다. exploration을 통해 더 좋은 action을 찾아내고 그를 더 많이 활용할 수 있는 기회가 생길 수 있기 때문이다. 

이는 위의 k-armed problem에 빗대어 생각해볼 수 있다. 슬롯머신마다 reward가 다른데 한번에 한 슬롯머신의 reward만 확인할 수 있기 때문에, 최대의 reward를 주는 슬롯머신을 탐색해야 한다. 랜덤하게 몇개의 슬롯을 당겨본 후 그 지식을 토대로 greedy한 선택을 할 수도 있고 \(exploitation\) 더 많은 정보를 얻기 위해 새로운 arm을 선택할 수도 있다.\(exploration\) 

* exploitation이 크면, 다른 더 좋은 reward를 가진 슬롯을 선택할 기회가 없어진다.
* exploration이 크면, 충분한 정보를 가지고 있더라도 정보를 더 얻기 위한 불필요한 비용이 발생한다.

결국 시간은 제한되어 있기 때문에 적절히 exploitation과 exploration간 적절한 균형을 유지해야 한다. 이 균형을 찾기 위한 것이 multi-armed problem의 핵심이다.

## Action-value Methods



## The 10-armed Testbed



## Incremental Implementation



## Tracking a Non-stationary Problem



## Optimistic Initial Values



## Upper-Confidence-Bound Action Selection



## Gradient Bandit Algorithms



## Associative Search





