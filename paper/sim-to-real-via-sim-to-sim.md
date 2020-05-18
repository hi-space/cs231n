---
description: >-
  Sim-to-Real via Sim-to-Sim: Data-efficient Robotic Grasping via
  Randomized-to-Canonical Adaptation Networks
---

# Sim-to-Real via Sim-to-Sim

* 현실에서 얻을 수 없는 데이터들을 모으고 해당 데이터로 training 시키기 위해서 labelled 된 simulation data를 사용한다. Simulation에서 training 하고 real world로 transfer 하면서 real 데이터 수집에 대한 cost를 절약할 수 있다. 하지만 두 도메인간 visual, physical 차이로 인해 transfer가 쉽게 가능하진 않다.
* Domain adaptation은 reality gap을 줄이기 위해 제안되었지만, unlabeled 된 real data를 상당히 많이 필요로 한다. 물론 real data에 일일이 annotation 하는 작업은 줄일 수 있지만 여전히 real data를 모으는데 cost가 발생한다.
  * Feature level adaptation : domain invariant 한 feature들을 학습한다
  * Pixel level adaptation : source 이미지가 target 이미지처럼 보이도록 restyling
* Domain Randomization은 동일한 이미지에서 다양한 변형을 줄 수 있기 때문에 다양한 feature 들을 학습할 수 있다. random한 texture, lighting, camera position을 통해 네트워크가 도메인 차이에 영향을 받지 않고 real world에 적용할 수 있게 한다. 하지만 이는 필요이상으로 문제를 어렵게 만들 수도 있다.
* 그래서 본 논문에서는 randomized 된 rendered image를 equivalent non-randomized 한, 즉 canonical version으로 바꾸도록 learning 하는 방식을 제안한다. 그리고 real image를 canonical sim image로 바꾸도록 변환하도록 한다. 

![](../.gitbook/assets/image%20%2847%29.png)

* randomized 한 simulation image 에서 canonical simulation 이미지를 변환하도록 하고 이 데이터를 agent를 training 시키도록 generator를 학습시킨다. 그 다음 real image를 canonical simulation 이미지로 변환하고 agent의 sim-to-real transfer가 되도록 한다. 
* Randomized 한 image를 non-randomized image 로 만드는 작업은 image-to-image translation 문제라고 볼 수 있다.
* 이 논문에서 제안한 Randomized to Canonical Adaptation Networks \(RCAN\)은 Domain Adaptation과 달리 real data가 필요하지 않다.
* GraspGAN 은 pixel-level adaptation, feature-level adaptation을 결합하여 학습에 필요한 real data 양을 줄였다. 하지만 역시 상당햔 양의 unlabelled real data를 필요로 한다.

![](../.gitbook/assets/image%20%28319%29.png)

* RCAN은 image-conditioned generative adversarial network \(cGAN\)으로 구성된다. cGAN은 randomized된 simulation 환경을 \(b\)와 같이 non-randomized한 canonical image를 얻게한다. training 된 cGAN은 반대로 real image 를 canonical 한 이미지로 변경할 수도 있다. 
* randomized simulation domain / canonical simulation domain / real-world domain 3가지 형태의 도메인이 존재한다.

$$
D = ((x_s, x_c, m_c, d_c)_j)^N_{j=1}
$$

* N :  training sample 갯수
* $$x_s $$ : randomization simulation domain의 RGB 이미지 \(source\)
* $$x_c $$ : canonical domain 의 RGB 이미지 \(target\) - semantic content 라고 보면 된다.
* $$m_c $$: segmentation mask
* $$d_c $$: depth image
* segmentation mask와 depth image는 training 할 때만 사용된다. RCAN generator 를 통해 input image를 넣으면 그에 해당하는 canonical RGB, Segmentation Mask, Depth image 가 생성된다.

![](../.gitbook/assets/image%20%2816%29.png)

