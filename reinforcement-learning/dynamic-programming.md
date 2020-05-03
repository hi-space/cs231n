# Planning by Dynamic Programming

## Introduction

‌

Planning은 model-based. 환경에 대해 이미 알고 있을 때 푸는 방법

### Whit is Dynamic Programming

‌

복잡한 문제를 푸는 하나의 방법론

‌

큰 문제들을 sub problem 들로 나누고, 작은 문제들에 대해 솔루션을 풀고 그 솔루션들을 모은다.

‌

Dynamic : 문제에서 sequential 하거나 temporal 한 컴포넌트

‌

Programming : program을 최적화 하는 것

### Requiremetns

작은 문제들로 나눠줘야 한다.

sub problem을 푼 결과를 cache로 저장해두고 있따가 나중에 다시 사용할 수 있어야 한다.

MDP가 이 위의 조건들을 만족시킨다. 그래서 Dynamic Programming 을 사용하기 적합하다.

recursive 한게 subproblem으로 나뉘는것을 말한다.

### Planning by Dynamic Programming

환경에 대해 모든 것을 알고 있어야 한다. Dynamic Programming은 Plannig에 쓰인다.

planning은 prediction과 control을 풀어야 한다.

predictino - value function을 찾는 것. MDP와 policy가 주어졌을 때 value function이 어떻게 되는 지 찾는 문제.

control - optimal policy를 찾는 것. MDP만 있을 때 optimal policy와 optimal value function을 찾아야하는 문제

policy evaluatino은 predicitno 문제, policy iteration / value iteraiotn은 control 문제

## Policy Evaluation

### Iterative Policy Evaluation

problem : policy를 따랐을 때 value function을 찾는 문제.

iterative 하게 bellman expectiation 문제를 풀어나간다.

backup?\) cache와 비슷한 개념.

* synchronous backup : 모든 state에 대해 한번씩 업데이트

v1→v2→v3 .. → v\_{\pi}

를 반복하다보면 수렴하게 된다.

normal state는 1~14, 1개는 terminal state

policy는 random policy







policy를 evaluation하기만 했는데 평가된 value에서 greedy하게 움직이다보면 optimal policy를 찾아낼 수 있다. 이상한 policy에 대해서 evaluaiton만 했는데 optimal policy를 찾았어!!!

## Policy Iteration‌

### How to Improve a Plicy

* Evaluate the policy \pi : value function을 찾기 위한 policy
* Improve the policy : 평가하는 value function에 의해 greedy 하게 움직이는 policy

small gridworld 에서는 policy를 improve 해서 optimal한 policy를 찾았다. 했다.





evaluation, improve를 반복하다보면 optimal policy를 찾을 수 있게 된다.

### Policy Improvement

policy improve 를 해서 greedy 하게 action을 취하면 정말 개선될까?

s 에 있을 때 파이를 따라서 value function을 따른 값 = s에서 파이가 골라준 action을 해서 value를 따라가는 값

max 어쩌구 → greedy 한 선택

bellman equation으로 푼거

언젠까는 더이상 improve되지 않는 상황이 온다

어떤 알고리즘을 써도 evaluaitnon이 되고 imporve 가 된다.

## Value Iteration‌

### Principle of Optimality

o첫 action이 optimal 해야하고, 그 optimal 한 action을 해서 다음 state를 따르고 난 후에는 optimal policy를 따른다.

### Determinisic Value Iteration

subproblem의 v\*\(s'\) 을 알 수 있다면, s 에서의 최적도 이 식을 이용해 알 수 있다.

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

