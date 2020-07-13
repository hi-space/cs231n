---
description: Asynchronous Methods for Deep Reinforcement Learning
---

# A3C

DQN은 Value Function Approximation 기반의 논문으로, Replay Memory를 통해 데이터 간의 correlation 줄이는 방식으로 좋을 결과를 냈었다. A3C는 Policy Gradient 기반의 논문으로,  Replay Memory 대신 Multiple Agent를 이용해 correlation을 줄이고 병렬적인 update를 통해 네트워크를 Scale up 한다. SOTA를 갈아치울 만큼 엄청난 성능을 보인 논문이였다.

> * sample 사이의 correlation을 비동기 업데이트로 해결 \(replay memory 대체\)
> * policy gradient 알고리즘 사용 가능 \(Actor-Critic\)
> * 상대적으로 빠른 학습 속도 \(Multiple Agent\)

![](../.gitbook/assets/image%20%28448%29.png)

각 Worker \(actor\) 들이 각각의 Environment 에서 경험을 쌓고 gradient를 계산하다가, 특정 iteration 마다 global network에게 얼만큼 업데이트 해야 하는지 비동기적으로 올려주게 되고, global network는 여러 Worker 들에게 받은 gradient 값들을 통해 얻어진 parameter 들을 worker로 내려주게 된다.\(value function 학습\) 각각의 worker가 병렬적으로 돌아가고 있기 때문에 학습 데이터 간 correlation이 줄어들고 학습속도가 super linear 하게 된다. 이 방법은 policy gradient 외의 다른 RL 알고리즘에서도 stable 한 결과를 보였다.

논문에서는 하나의 machine에 multi thread로 구현했다. agent의 최대 갯수는 thread 의 수 만큼이였지만 충분히 좋은 성능을 냈다. 각 action learner 가 서로 다른 exploration policy를 갖도록 했는데, 이 부분이 성능과 robustness에 영향을 줬다.

> **super linear**: worker의 갯수가 2개 일 때 속도가 2배가 되는 것을 linear 하다고 표현할 수 있다. worker를 2개 사용했지만 속도가 2배 이상으로 빨라진 것을 super linear 라고 한다.
>
> **actor**: environment와 agent가 상호작용 하며 경험을 쌓음
>
> **learner**: 쌓인 경험으로 학습을 시키는 애

논문에서는 4가지의 RL Framework 를 제안했다. Off-Policy \(One-step Q-Learning, N-step Q-Learning\)과 On-Policy \(One-step SARSA, Actor-Critic\) 모두에 실험을 해봤는데 결과적으로는 Asynchronous Advantage Actor-Critic이 가장 좋은 성능을 보였고 이를 A3C 라고 한다.

* Asynchronous One-step Q-Learning
* Asynchronous N-step Q-Learning
* Asynchronous One-step SARSA 
* Asynchronous Advantage Actor-Critic

### Advantage Actor Critic \(A2C\)

![A2C](../.gitbook/assets/image%20%28450%29.png)

* `Actor` : Policy를 통해 Action을 취하는 Agent
* `Critic` : Value Function을 통해 현재 상태를 Evaluation

A2C는 A3C의 Synchronous하고 Deterministic 한 버전이라고 보면 된다. 

![A3C vs A2C](../.gitbook/assets/image%20%28447%29.png)

