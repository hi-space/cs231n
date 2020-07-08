---
description: Human-level control through deep reinforcement learning
---

# DQN

## Introduction

이전의 RL 문제에서는 vision이나 speech 같은 high-dimensional한 데이터들에 적용해 agent를 직접적으로 학습시키기가 어려웠었다. 그래서 주로 hand-crafted 한 feature에 대해서만 문제를 적용해왔다. 하지만 computer vision이나 speech recognition 분야에서 raw sensor data를 이용한 CNN, RNN 문제들이 유의미한 결과를 얻어내기 시작했고 RL로도 이러한 센서 데이터를 이용한 문제를 풀어보려고 한다.

이런 문제를 적용할 때 기존 DL 문제들과 달리 RL 에만 존재하는 문제들이 있다.

1. sparse하고 noisy 한데다 delayed된 reward 값을 이용해 학습해야한다. 
2. 기존의 DL 문제들은 data sample 각각이 독립적이였지만 \(i.i.d.\) RL 에서는 연속된 state 간의 correlation이 커서 학습이 어렵다.

이 논문에 목표는 raw video data를 이용해 복잡한 RL 환경에서 성공적으로 control 문제를 해결하는 것이다. 논문에서는 복잡한 데이터를 직접적으로 input으로 사용하면서, 하나의 agent가 여러가지의 문제\(Atari 2600 게임\)를 해결하게 된다. 여기에서 agent를 학습시키기 위해 제안된 방법들은 지금도 RL 문제를 푸는데 굉장히 많이 사용되고 있다.  

Experience Replay Mechanism 과거의 transition 들을 랜덤으로 sampling 한 데이터들을 학습 데이터로 사용하게 된다.

## Deep Reinforcement Learning

Deep Neural Network 와 RL을 연결해서, 문제에 대한 추가적인 정보 없이도 RGB 이미지만을 training data로 사용해 stochastic gradient를 update 하며 학습하는 것이 목표이다. 

이전의 RL 에서는 environment와 interaction 하며 얻어진 on-policy의 샘플들을 통해 Q-network의 parameter를 업데이트 했었다. 이 경우 sample에 대해 의존성이 크기 때문에 policy 변화에 따른 Q-value의 변화량이 너무 커서 converge 하지 못하고 oscillate 하기 쉬웠다. 이 문제를 해결하기 위해 아래와 같은 두가지 방법을 사용한다.

### Experience Replay

$$e_t = (s_t, a_t, r_t, s_{t+1})$$

각 time step 마다 얻어진 sample \(경험\) $$e_t$$을 위와 같이 tuple로 정의하여 N 사이즈의 Replay Buffer라는 곳에 FIFO로 저장한다. 그리고 Replay Buffer에 어느정도 경험이 누적되고 난 후에는 Replay Buffer의 sample들을 uniform random으로 뽑아 mini-batch를 만들고 그 데이터를 통해 network를 학습시킨다. 추출한 mini-batch 샘플 간에는 게임과의 연관성이 없을 수도 있고 연속적이지 않은 데이터들이기 때문에 데이터 간 correlation을 없앨 수 있다. 또한 누적되어 있는 경험들을 재사용할 수도 있기 때문에 data efficient 하기도 하다. 

Experience Replay를 사용할 때에는 현재의 parameter와 update 하는 샘플들과 다르기 때문에 반드시 off-policy로 learning 해야 한다.

![pseudo code](../../.gitbook/assets/image%20%28448%29.png)

### Freeze target Q-network

fixed Q-targets 방법이다. TD target을 계산할 때 parameter들을 고정시켜놓고 학습시키는 방법이다. target network 두개를 둬 하나의 network는 Q 값만 update 하게 한 후 특정 iteration 마다 다른 network에게 parameter 값을 복사해준다. TD target이 바라보는 방향을 일정 iteration 만큼 고정시켜놓음으로써 stable 하게 학습되도록 하는 것이다.



추가적으로 마이너한 이슈로는 reward와 Q-value의 값이 너무 커질 수도 있기 때문에 stable한 SGD 업데이트가 어렵다는 점이 있었다. 이 경우는 reward를 clip하거나 network를 특정 범위로 normalize 하는 방식으로 처리했다. 

## Model Architecture

![Pre-processing](../../.gitbook/assets/image%20%28449%29.png)

Atari 게임의 RGB 이미지를 넣기 전에 위와 같이 preprocessing 과정을 거친다. 그리고 input data를 넣을 때에는 4개의 history를 stack으로 쌓아서 네트워크에 넣어주게 된다. 

![Model Architecture](../../.gitbook/assets/image%20%28447%29.png)

3개의 CNN과 2개의 FC 를 이어서 모델을 설계했다.

## Reference

{% embed url="https://sumniya.tistory.com/18" %}

{% embed url="http://sanghyukchun.github.io/90/" %}

{% embed url="https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf" %}



