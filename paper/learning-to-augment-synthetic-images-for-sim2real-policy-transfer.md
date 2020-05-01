# Learning to Augment Synthetic Images for Sim2Real Policy Transfer

image understanding 문제를 해결하기 위해서는 domain-specific 한 visual data가 상당량 필요하다. 이러한데이터들을 얻을 수는 있으나 한계가 있기 때문에 scalability 가 떨어진다. 그래서 본 논문에서는 로봇의 manipulation policy을 simulation 환경을 통해 학습하고자 한다. simulation 을 이용함으로써 scalability가 늘어나고 training도 가능하지만, real 환경과 synthetic data 간에 domain gap이 존재한다. 이전에는 domain randomization과 random transformation을 통해 synthetic image를 확장시켰고, 이 방법들을 따르고 있다. 

이 논문의 contribution은  sim2real transfer를 할 때 \(domain-independent한 policy를 learning 하기 위해서\) augmentation 전략을 optimize 하는 것이다. object localization을 이용해 depth image augmentation을 위해 효율적으로 search 하도록 디자인한다. 

### Introduction

최근에는 학습 기반 인식 뿐만 아니라 vision과 제어를 결합한 다양한 문제들이 제안되고 있다. \(ex, object detection, image segmentation, human estimation\) 그러나 로봇을 학습시키는 것은 많은 시도들이 필요해서 complex task, environments 를 확장하는데 어려움이 있다. 여기서 physics simulator와 graphics engine 은 병렬화, multiple environments 등을 위해 좋은 대안으로 사용될 수 있다. reality gap을 줄이기 위해 domain randomization 을 사용하고 있고 꽤 좋은 결과를 얻을 수 있었지만 optimality, generality 문제는 여전히 남아 있다. 

본 연구에서는 domain randomization 방법을 따르고 sim2real transfer를 최적화하는 방법을 learning 하는 것을 제안한다. 



![](../.gitbook/assets/image%20%288%29.png)

Synthetic depth map 에서 random transformations를 적용하여 policy training 시키고 실제 로봇의 depth map 정보로 부터 training 한 policy를 테스트한 결과물이다.

![](../.gitbook/assets/image%20%2892%29.png)

depth image augmentation 의 policy-independent한 learning 이 이 논문의 contribution 이다. augmentation의 결과 시퀀스는 manipulation의 policy를 학습하는 synthetic depth image에 적용된다.  학습된 policy는 실제 image로 fine tuning 하지 않고 실제 로봇에 그대로 적용된다. 

다양한 synthetic depth image 들을 만들어내고 거기에 random으로 변환 시퀀스를 적용한 후 객체 위치를 추정하는 CNN regressor를 훈련시킨다. 그 후에는 실제 이미지에서 CNN의 위치 예측을 validation 하면서 현재 random으로 적용했던 그 변환 시퀀스의 매개변수를 검증한다.  \(Monte-Carlo Tree Search 방식\)

Domain Randomization은 심플하고 효과가 좋아서 자주 사용됐지만 이건 RGB image 였을 때의 이야기이다. Depth image 는 어떤것을 Randomize 해야하는 지 아직 명확하지 않다. 그래서 여기에선 학습 기반으로 어떤 randomization을 선택해야 할 지 접근 방식을 탐색하고, 그를 통해 visual data의 gap을 줄이고자 한다.

Depth image에 random noise를 생성하기 위해 실제 센서의 noise pattern을 넣거나, 누락된 정보들을 보정하여 depth image 정보를 확대할 수 있다. 

자동으로 가장 좋은 augmentation을 찾기 위해서 RL을 사용하거나 수많은 GPU 자원이 필요하다. 그래서 Monte Carlo Tree Search를 사용해서 사전에 기록된 real image에서의 객체 위치를 예측해서 proxy task 내에서 최적화를 한다. \(시뮬레이션의 렌더링 파이프라인을 학습하는 아이디어의 논문도 있음 - Learning to Simulate\)

![](../.gitbook/assets/image%20%28265%29.png)

![](../.gitbook/assets/image%20%28234%29.png)



